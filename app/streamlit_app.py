"""
ParkSentinel — Streamlit Dashboard
Day 4: Full production dashboard

Run with:
    streamlit run app/streamlit_app.py

Pages:
  1. Overview Map      — H3 hex choropleth + violation heatmap
  2. Enforcement Queue — live ranked patrol list for current time
  3. Zone Deep Dive    — per-zone stats, hourly chart, top violations
  4. Patrol Scheduler  — full week calendar
  5. AI Assistant      — Gemini 2.5 Flash chat interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import folium
import h3
import json
import ast
import os
import datetime
import warnings
from pathlib import Path
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS    = ROOT / "models"

TIER_COLORS = {
    "Critical" : "#e74c3c",
    "High"     : "#e67e22",
    "Medium"   : "#f1c40f",
    "Low"      : "#2ecc71",
    "Very Low" : "#3498db",
    "nan"      : "#95a5a6",
}

st.set_page_config(
    page_title="ParkSentinel — Bengaluru",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: #1e2130; border-radius: 10px;
        padding: 16px 20px; margin: 6px 0;
        border-left: 4px solid #e74c3c;
    }
    .metric-card h3 { color: #e74c3c; margin: 0; font-size: 1.6rem; }
    .metric-card p  { color: #aaa; margin: 4px 0 0 0; font-size: 0.85rem; }
    .tier-critical { color: #e74c3c; font-weight: bold; }
    .tier-high     { color: #e67e22; font-weight: bold; }
    .tier-medium   { color: #f1c40f; font-weight: bold; }
    .stDataFrame   { font-size: 0.82rem; }
    div[data-testid="stSidebarContent"] { background: #1a1d27; }
</style>
""", unsafe_allow_html=True)


# ── Data Loading (cached) ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading violation data...")
def load_data():
    hex_df = pd.read_parquet(PROCESSED / "h3_priority_scores.parquet")

    zp = pd.read_parquet(PROCESSED / "zone_profiles_day3.parquet")
    for col in ["top_violations","top_vehicles","hourly_counts","daily_counts"]:
        if col in zp.columns:
            zp[col] = zp[col].apply(lambda v: json.loads(v) if isinstance(v,str) else v)

    cal  = pd.read_parquet(PROCESSED / "patrol_calendar.parquet")
    vmap = pd.read_parquet(PROCESSED / "violations_with_h3.parquet")

    return hex_df, zp, cal, vmap


@st.cache_resource(show_spinner="Connecting to Gemini...")
def load_gemini():
    key = os.getenv("GOOGLE_API_KEY","")
    if not key:
        return None, None
    genai.configure(api_key=key)
    flash = genai.GenerativeModel("gemini-2.5-flash")
    pro   = genai.GenerativeModel("gemini-2.5-pro")
    return flash, pro


hex_df, zone_df, cal_df, vmap_df = load_data()
gemini_flash, gemini_pro         = load_gemini()


# ── Helpers ───────────────────────────────────────────────────────────────
def get_ist():
    return datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=5, minutes=30)))

def get_patrol_recs(hour, day_name, top_n=10):
    recs = []
    for _, row in zone_df.iterrows():
        base    = row["ensemble_score"]
        hourly  = row["hourly_counts"]
        daily   = row["daily_counts"]
        th      = max(sum(hourly.values()), 1)
        td      = max(sum(daily.values()), 1)
        tr      = hourly.get(hour, hourly.get(str(hour), 0)) / th
        dr      = daily.get(day_name, 0) / td
        adj     = base * (1 + 0.4*tr + 0.2*dr)
        recs.append({
            "Station"     : row["police_station"],
            "Junction"    : row["top_junction"][:35] if row["top_junction"] else "—",
            "Adj Score"   : round(adj, 1),
            "Base Score"  : round(base, 1),
            "Tier"        : row["priority_tier"],
            "Peak Hour"   : f"{row['peak_hour']}:00",
            "Peak Day"    : row["peak_day"],
            "Top Offense" : (", ".join(row["top_violations"][:1])
                             if row["top_violations"] else "—"),
            "Top Vehicle" : (", ".join(row["top_vehicles"][:1])
                             if row["top_vehicles"] else "—"),
            "lat"         : row["lat"],
            "lon"         : row["lon"],
            "h3_id"       : row["h3_id"],
        })
    return (pd.DataFrame(recs)
              .sort_values("Adj Score", ascending=False)
              .head(top_n)
              .reset_index(drop=True))


def build_context(top_n=10):
    zones = zone_df.nlargest(top_n,"ensemble_score")
    lines = ["=== BENGALURU PARKING VIOLATION DATA ===\n"]
    for i,(_, r) in enumerate(zones.iterrows(), 1):
        lines.append(
            f"Zone #{i} | Station: {r['police_station']} | "
            f"Junction: {r['top_junction']}\n"
            f"  Score: {r['ensemble_score']:.1f}/100 | Tier: {r['priority_tier']}\n"
            f"  Violations: {r['violation_count']:,} | Peak: {r['peak_hour']}:00 on {r['peak_day']}\n"
            f"  Morning patrol: {r['best_morning_start']}:00–{r['best_morning_start']+2}:00 | "
            f"Evening: {r['best_evening_start']}:00–{r['best_evening_start']+2}:00\n"
            f"  Weekend: {r['weekend_ratio']*100:.0f}% | Junction: {r['near_junction_ratio']*100:.0f}%\n"
            f"  Violations: {', '.join(r['top_violations']) if r['top_violations'] else 'N/A'}\n"
            f"  Vehicles: {', '.join(r['top_vehicles']) if r['top_vehicles'] else 'N/A'}\n"
            f"  Severity: {r['severity_mean']:.2f}/5\n"
        )
    return "\n".join(lines)


SYSTEM_PROMPT = """You are ParkSentinel, AI assistant for Bengaluru Traffic Police.
Help inspectors understand parking hotspots and plan enforcement.
Always cite specific numbers from the data. Keep answers under 200 words.
Be operational: tell the inspector WHAT to do and WHEN."""


def ask_gemini(question, use_pro=False):
    if not gemini_flash:
        return "⚠️ Gemini not connected — check GOOGLE_API_KEY in .env"
    ctx    = build_context(top_n=10)
    prompt = f"{SYSTEM_PROMPT}\n\n{ctx}\n\nInspector: {question}\n\nParkSentinel:"
    model  = gemini_pro if use_pro else gemini_flash
    return model.generate_content(prompt).text


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-light.png", width=60)
    st.title("ParkSentinel")
    st.caption("Bengaluru Traffic Enforcement Intelligence")
    st.markdown("---")

    page = st.radio("Navigate", [
        "🗺️  Overview Map",
        "🚨  Enforcement Queue",
        "🔍  Zone Deep Dive",
        "📅  Patrol Scheduler",
        "🤖  AI Assistant",
    ])
    st.markdown("---")

    # Live clock
    ist = get_ist()
    st.markdown(f"**IST:** `{ist.strftime('%H:%M:%S')}  {ist.strftime('%A')}`")
    st.markdown(f"**Date:** `{ist.strftime('%d %b %Y')}`")
    st.markdown("---")

    # KPIs
    total_v   = int(hex_df["violation_count"].sum())
    n_crit    = int((hex_df["final_priority_tier"]=="Critical").sum())
    n_high    = int((hex_df["final_priority_tier"]=="High").sum())
    top_score = float(hex_df["ensemble_score"].max())

    st.markdown(f"""
    <div class="metric-card">
      <h3>{total_v:,}</h3><p>Total Violations (Jan–May)</p>
    </div>
    <div class="metric-card">
      <h3>{n_crit}</h3><p>Critical Zones</p>
    </div>
    <div class="metric-card">
      <h3>{n_high}</h3><p>High Priority Zones</p>
    </div>
    <div class="metric-card">
      <h3>{top_score:.1f}</h3><p>Highest Zone Score</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW MAP
# ═══════════════════════════════════════════════════════════════════════════
if page.startswith("🗺️"):
    st.title("🗺️ Bengaluru Parking Violation Heatmap")
    st.caption("H3 hexagonal priority zones + violation density heatmap")

    col1, col2, col3 = st.columns(3)
    with col1:
        min_score = st.slider("Min Priority Score", 0, 80, 25, 5)
    with col2:
        map_style = st.selectbox("Map Style",
            ["CartoDB dark_matter","CartoDB positron","OpenStreetMap"])
    with col3:
        show_heat = st.checkbox("Show Violation Heatmap", value=True)

    m = folium.Map(location=[12.97,77.59], zoom_start=12, tiles=map_style)

    # Violation heatmap layer
    if show_heat:
        sample = vmap_df.sample(min(15_000, len(vmap_df)), random_state=42)
        heat_data = list(zip(sample["latitude"], sample["longitude"],
                              sample["severity_score"].fillna(1)))
        HeatMap(heat_data, radius=8, blur=10, max_zoom=14,
                gradient={"0.2":"blue","0.5":"lime","0.8":"orange","1.0":"red"},
                name="Violation Density").add_to(m)

    # H3 hex polygons
    drawn = 0
    for _, row in hex_df.iterrows():
        score = row["ensemble_score"]
        if score < min_score: continue
        tier  = str(row["final_priority_tier"])
        color = TIER_COLORS.get(tier, "#95a5a6")

        boundary  = h3.cell_to_boundary(row["h3_id"])
        poly_coords = [[lat, lon] for lat, lon in boundary]

        popup_html = (
            f"<b>{tier}</b> — Score: {score:.1f}<br>"
            f"<b>Station:</b> {row['police_station']}<br>"
            f"<b>Junction:</b> {row['top_junction']}<br>"
            f"<b>Violations:</b> {int(row['violation_count']):,}<br>"
            f"<b>Severity avg:</b> {row['severity_mean']:.2f}<br>"
            f"<b>Near junction:</b> {row['near_junction_ratio']*100:.0f}%<br>"
            f"<b>Peak hour:</b> {row['peak_hour_ratio']*100:.0f}% in rush hours"
        )
        folium.Polygon(
            locations=poly_coords, color=color, fill=True,
            fill_color=color, fill_opacity=0.55, weight=1,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{tier} ({score:.0f})"
        ).add_to(m)
        drawn += 1

    # Top 10 markers
    top10 = hex_df.nlargest(10,"ensemble_score")
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        folium.Marker(
            location=[row["lat"], row["lon"]],
            tooltip=f"#{rank} {row['police_station'][:25]} ({row['ensemble_score']:.0f})",
            popup=f"#{rank} CRITICAL<br>{row['police_station']}<br>Score: {row['ensemble_score']:.1f}",
            icon=folium.Icon(color="red", icon="exclamation-sign")
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=1200, height=600)
    st.caption(f"Showing {drawn} hex zones with score ≥ {min_score}. Click any hex for details.")

    # Legend
    st.markdown("**Legend:**  " + "  ".join(
        f'<span style="color:{c}">■</span> {t}'
        for t,c in TIER_COLORS.items() if t != "nan"
    ), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — ENFORCEMENT QUEUE
# ═══════════════════════════════════════════════════════════════════════════
elif page.startswith("🚨"):
    st.title("🚨 Live Enforcement Queue")
    st.caption("Zones ranked by time-adjusted priority score for current IST time")

    ist  = get_ist()
    col1, col2, col3 = st.columns(3)
    with col1:
        query_hour = st.slider("Hour (IST)", 0, 23, ist.hour)
    with col2:
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        query_day  = st.selectbox("Day", days, index=ist.weekday())
    with col3:
        top_n = st.slider("Show top N zones", 5, 20, 10)

    recs = get_patrol_recs(query_hour, query_day, top_n=top_n)

    # Colour tier column
    def style_tier(val):
        colors = {"Critical":"#e74c3c","High":"#e67e22",
                  "Medium":"#f1c40f","Low":"#2ecc71","Very Low":"#3498db"}
        return f"color: {colors.get(val,'white')}; font-weight: bold"

    st.markdown(f"#### Top {top_n} zones at **{query_hour:02d}:00** on **{query_day}**")

    display_cols = ["Station","Junction","Adj Score","Tier","Peak Hour","Peak Day",
                    "Top Offense","Top Vehicle"]
    styled = recs[display_cols].style.map(style_tier, subset=["Tier"])
    st.dataframe(styled, use_container_width=True, height=420)

    # Mini map of top 5
    st.markdown("#### Top 5 Patrol Zones — Map")
    m2 = folium.Map(location=[12.97,77.59], zoom_start=12, tiles="CartoDB dark_matter")
    for i, (_, row) in enumerate(recs.head(5).iterrows(), 1):
        folium.Marker(
            location=[row["lat"], row["lon"]],
            tooltip=f"#{i} {row['Station']} (Score: {row['Adj Score']})",
            popup=f"<b>#{i} — {row['Station']}</b><br>Score: {row['Adj Score']}<br>Tier: {row['Tier']}",
            icon=folium.Icon(
                color=["red","orange","orange","blue","blue"][i-1],
                icon=str(i), prefix="fa"
            )
        ).add_to(m2)
    st_folium(m2, width=900, height=380)

    # Impact estimator
    st.markdown("---")
    st.markdown("#### 📏 Carriageway Impact Estimator")
    st.caption("Estimates road space freed if violations in selected zone are cleared.")
    sel_station = st.selectbox("Select zone", recs["Station"].tolist())
    sel_row     = recs[recs["Station"]==sel_station].iloc[0]
    sel_hex     = hex_df[hex_df["police_station"]==sel_station]
    if len(sel_hex) > 0:
        viol_count = int(sel_hex.iloc[0]["violation_count"])
        avg_veh_len = 4.5   # metres (car)
        lane_width  = 3.5   # metres
        freed_m     = viol_count * avg_veh_len
        freed_lanes = freed_m / lane_width
        c1, c2, c3 = st.columns(3)
        c1.metric("Violations in Zone", f"{viol_count:,}")
        c2.metric("Carriageway Freed", f"{freed_m:,.0f} m")
        c3.metric("Equivalent Lane-metres", f"{freed_lanes:,.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — ZONE DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════
elif page.startswith("🔍"):
    st.title("🔍 Zone Deep Dive")

    zone_options = zone_df.sort_values("ensemble_score", ascending=False)["police_station"].tolist()
    selected     = st.selectbox("Select Zone (by Police Station)", zone_options)
    row          = zone_df[zone_df["police_station"]==selected].iloc[0]

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Priority Score",   f"{row['ensemble_score']:.1f}")
    c2.metric("Tier",             row["priority_tier"])
    c3.metric("Total Violations", f"{row['violation_count']:,}")
    c4.metric("Peak Hour",        f"{row['peak_hour']}:00 IST")
    c5.metric("Peak Day",         row["peak_day"])

    st.markdown(f"**Junction:** {row['top_junction']}  |  "
                f"**Morning patrol:** `{row['best_morning_start']}:00–{row['best_morning_start']+2}:00`  |  "
                f"**Evening patrol:** `{row['best_evening_start']}:00–{row['best_evening_start']+2}:00`")
    st.markdown(f"**Weekend share:** {row['weekend_ratio']*100:.0f}%  |  "
                f"**Near junction:** {row['near_junction_ratio']*100:.0f}%  |  "
                f"**Avg severity:** {row['severity_mean']:.2f}/5")
    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        # Hourly profile
        hourly = row["hourly_counts"]
        hours  = list(range(24))
        counts = [hourly.get(h, hourly.get(str(h), 0)) for h in hours]
        colors = ["#e74c3c" if h in list(range(7,11))+list(range(17,22))
                  else "#3498db" for h in hours]

        fig, ax = plt.subplots(figsize=(7,3.5))
        fig.patch.set_facecolor("#1e2130")
        ax.set_facecolor("#1e2130")
        ax.bar(hours, counts, color=colors, width=0.8)
        ax.set_xlabel("Hour (IST)", color="white")
        ax.set_ylabel("Violations", color="white")
        ax.set_title("Hourly Violation Profile  [Red=Peak]",
                     color="white", fontweight="bold")
        ax.tick_params(colors="white")
        ax.set_xticks([0,6,12,18,23])
        for spine in ax.spines.values(): spine.set_visible(False)
        st.pyplot(fig, use_container_width=True)

    with col_r:
        # Daily profile
        daily  = row["daily_counts"]
        day_order2 = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        d_counts = [daily.get(d, 0) for d in day_order2]
        d_colors = ["#e74c3c" if d in ["Saturday","Sunday"] else "#2ecc71"
                    for d in day_order2]

        fig2, ax2 = plt.subplots(figsize=(7,3.5))
        fig2.patch.set_facecolor("#1e2130")
        ax2.set_facecolor("#1e2130")
        ax2.bar([d[:3] for d in day_order2], d_counts, color=d_colors)
        ax2.set_xlabel("Day", color="white")
        ax2.set_ylabel("Violations", color="white")
        ax2.set_title("Day-of-Week Profile  [Red=Weekend]",
                      color="white", fontweight="bold")
        ax2.tick_params(colors="white")
        for spine in ax2.spines.values(): spine.set_visible(False)
        st.pyplot(fig2, use_container_width=True)

    # Violation & vehicle breakdown
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Top Violation Types**")
        for v in row["top_violations"]:
            st.markdown(f"- {v}")
    with col4:
        st.markdown("**Top Vehicle Types**")
        for v in row["top_vehicles"]:
            st.markdown(f"- {v}")

    # Mini map centred on zone
    st.markdown("---")
    st.markdown("**Zone Location**")
    m3 = folium.Map(location=[row["lat"],row["lon"]], zoom_start=15,
                    tiles="CartoDB dark_matter")
    folium.Marker(
        location=[row["lat"],row["lon"]],
        tooltip=selected,
        icon=folium.Icon(color="red", icon="exclamation-sign")
    ).add_to(m3)
    # Draw the hex
    boundary = h3.cell_to_boundary(row["h3_id"])
    folium.Polygon(
        locations=[[lat,lon] for lat,lon in boundary],
        color="#e74c3c", fill=True, fill_opacity=0.4, weight=2
    ).add_to(m3)
    st_folium(m3, width=900, height=350)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — PATROL SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════
elif page.startswith("📅"):
    st.title("📅 Weekly Patrol Scheduler")
    st.caption("Top patrol zone per time slot. Use this to assign officers for the week.")

    day_order3 = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    # Score heatmap
    pivot_score = cal_df[cal_df["rank"]==1].pivot_table(
        index="day", columns="time_slot", values="score", aggfunc="first"
    ).reindex(day_order3)

    fig, ax = plt.subplots(figsize=(13,5))
    fig.patch.set_facecolor("#1e2130"); ax.set_facecolor("#1e2130")
    sns.heatmap(pivot_score, cmap="YlOrRd", ax=ax, linewidths=0.5,
                annot=True, fmt=".0f", annot_kws={"size":10},
                cbar_kws={"label":"Priority Score"})
    ax.set_title("Patrol Priority Calendar — Score per Slot (Higher = More Urgent)",
                 color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.set_xlabel("Time Slot (IST)", color="white")
    ax.set_ylabel("Day of Week", color="white")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Zone name heatmap
    pivot_names = cal_df[cal_df["rank"]==1].pivot_table(
        index="day", columns="time_slot", values="station", aggfunc="first"
    ).reindex(day_order3)

    fig2, ax2 = plt.subplots(figsize=(13,5))
    fig2.patch.set_facecolor("#1e2130"); ax2.set_facecolor("#1e2130")
    annot = pivot_names.map(lambda x: x[:12] if isinstance(x,str) else "")
    sns.heatmap(pivot_score, cmap="Blues", ax=ax2, linewidths=0.5,
                annot=annot, fmt="", annot_kws={"size":8},
                cbar_kws={"label":"Score"})
    ax2.set_title("Patrol Assignment — Zone Name per Slot",
                  color="white", fontweight="bold")
    ax2.tick_params(colors="white")
    ax2.set_xlabel("Time Slot (IST)", color="white")
    ax2.set_ylabel("Day of Week", color="white")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

    # Drill-down table
    st.markdown("---")
    st.markdown("#### Drill Down")
    sel_day  = st.selectbox("Day", day_order3)
    sel_slot = st.selectbox("Time Slot", cal_df["time_slot"].unique())
    drill = cal_df[(cal_df["day"]==sel_day)&(cal_df["time_slot"]==sel_slot)].sort_values("rank")
    st.dataframe(drill[["rank","station","junction","score","tier"]],
                 use_container_width=True)

    # Download
    readable = pivot_names.fillna("—")
    csv = readable.to_csv()
    st.download_button("⬇️ Download Schedule CSV", csv,
                       "patrol_schedule.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — AI ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════
elif page.startswith("🤖"):
    st.title("🤖 ParkSentinel AI Assistant")
    st.caption("Powered by Gemini 2.5 Flash — grounded in your actual violation data")

    if not gemini_flash:
        st.error("Gemini not connected. Add GOOGLE_API_KEY to your .env file.")
        st.stop()

    col_l, col_r = st.columns([3,1])
    with col_r:
        use_pro = st.toggle("Use Gemini 2.5 Pro", value=False)
        model_name = "gemini-2.5-pro" if use_pro else "gemini-2.5-flash"
        st.caption(f"Model: `{model_name}`")
        st.markdown("---")
        st.markdown("**Quick questions:**")
        quick_qs = [
            "Where to patrol at 8 AM on Monday?",
            "Top 3 critical zones and why?",
            "Which vehicles cause worst congestion?",
            "Best weekend patrol strategy?",
            "Which zone is trending upward?",
        ]
        for q in quick_qs:
            if st.button(q, key=q):
                st.session_state["prefill"] = q

    with col_l:
        # Chat history
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Display history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"],
                avatar="👮" if msg["role"]=="user" else "🚦"):
                st.markdown(msg["content"])

        # Input
        prefill = st.session_state.pop("prefill", "")
        user_input = st.chat_input("Ask about Bengaluru parking violations...",
                                    key="chat_input") or prefill

        if user_input:
            # Show user message
            with st.chat_message("user", avatar="👮"):
                st.markdown(user_input)
            st.session_state["messages"].append(
                {"role":"user","content":user_input})

            # Get Gemini response
            with st.chat_message("assistant", avatar="🚦"):
                with st.spinner("Analysing violation data..."):
                    response = ask_gemini(user_input, use_pro=use_pro)
                st.markdown(response)

            st.session_state["messages"].append(
                {"role":"assistant","content":response})

        if st.button("🗑️ Clear Chat"):
            st.session_state["messages"] = []
            st.rerun()
