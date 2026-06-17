"""
ParkSentinel — Streamlit Dashboard (REVISED)
============================================
Run: streamlit run app/streamlit_app.py

Pages:
  1. Overview Map      — H3 hex choropleth, violation heatmap, toggles
  2. Enforcement Queue — stratified missions + full ranked queue, slider re-sorts
  3. Zone Deep Dive    — ALL zones, Plotly charts, no duplicates
  4. Patrol Scheduler  — stratified calendar + drill-down (top 15 per slot)
  5. AI Assistant      — full-city context, Flash/Pro toggle, quick buttons
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
import h3
import json
import ast
import os
import datetime
import warnings
import re
import threading
import time
from pathlib import Path
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

# ── Constants ──────────────────────────────────────────────────────────────
TIER_COLORS = {
    "Critical" : "#e74c3c",
    "High"     : "#e67e22",
    "Medium"   : "#f1c40f",
    "Low"      : "#2ecc71",
    "Very Low" : "#3498db",
}
TIER_ORDER  = ["Critical","High","Medium","Low","Very Low"]
DAY_ORDER   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
SLOTS       = [(6,"06–09"),(9,"09–12"),(12,"12–15"),(15,"15–18"),(18,"18–21"),(21,"21–24")]

st.set_page_config(
    page_title="ParkSentinel — Bengaluru",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"]          { background: #1a1d27; }
.mission-card {
    border-radius:10px; padding:14px 18px; margin:6px 0;
    border-left:5px solid;
}
.firefighter  { border-color:#e74c3c; background:#1f1217; }
.preventative { border-color:#f1c40f; background:#1f1e10; }
.wildcard     { border-color:#3498db; background:#101820; }
.mission-card h4 { margin:0 0 6px 0; font-size:1rem; }
.mission-card p  { margin:2px 0; font-size:0.82rem; color:#ccc; }
.kpi-row { display:flex; gap:12px; margin-bottom:10px; }
.kpi-box {
    flex:1; background:#1e2130; border-radius:8px;
    padding:12px 16px; border-top:3px solid #e74c3c;
}
.kpi-box h3 { margin:0; font-size:1.5rem; color:#fff; }
.kpi-box p  { margin:4px 0 0; font-size:0.78rem; color:#aaa; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading zone profiles…")
def load_zone_profiles():
    zp = pd.read_parquet(PROCESSED / "zone_profiles_day3.parquet")
    for col in ["top_violations","top_vehicles","hourly_counts","daily_counts"]:
        if col in zp.columns:
            zp[col] = zp[col].apply(
                lambda v: json.loads(v) if isinstance(v, str) else v
            )
    return zp

@st.cache_data(show_spinner="Loading hex map data…")
def load_hex():
    path = PROCESSED / "h3_priority_scores_enriched.parquet"
    if not path.exists():
        path = PROCESSED / "h3_priority_scores.parquet"
    return pd.read_parquet(path)

@st.cache_data(show_spinner="Loading patrol calendar…")
def load_calendar():
    return pd.read_parquet(PROCESSED / "patrol_calendar.parquet")

@st.cache_data(show_spinner="Loading violation map layer…")
def load_vmap():
    return pd.read_parquet(PROCESSED / "violations_with_h3.parquet")

@st.cache_resource(show_spinner="Connecting to Gemini…")
def load_gemini():
    key = os.getenv("GOOGLE_API_KEY","")
    if not key:
        return None, None
    genai.configure(api_key=key)
    return (genai.GenerativeModel("gemini-2.5-flash"),
            genai.GenerativeModel("gemini-2.5-pro"))

zone_df       = load_zone_profiles()
hex_df        = load_hex()
calendar_df   = load_calendar()
vmap_df       = load_vmap()
gemini_flash, gemini_pro = load_gemini()


# ══════════════════════════════════════════════════════════════════════════
# SCHEDULER HELPERS
# ══════════════════════════════════════════════════════════════════════════
def compute_adj_score(row, hour, day_name, boost=3.0):
    hc      = row["hourly_counts"]
    dc      = row["daily_counts"]
    total_h = max(sum(hc.values()), 1)
    total_d = max(sum(dc.values()), 1)
    t_rel   = hc.get(hour, hc.get(str(hour), 0)) / total_h
    d_rel   = dc.get(day_name, dc.get(str(day_name), 0)) / total_d
    return row["ensemble_score"] * (1 + boost * t_rel + 0.5 * d_rel)

@st.cache_data(show_spinner=False)
def get_scored_df(hour, day_name):
    """Return full zone_df with time-adjusted scores — cached per (hour, day)."""
    rows = []
    for _, row in zone_df.iterrows():
        adj = compute_adj_score(row, hour, day_name)
        rows.append({**row.to_dict(), "adj_score": round(adj, 1)})
    return pd.DataFrame(rows).sort_values("adj_score", ascending=False)

def get_stratified_missions(hour, day_name):
    scored = get_scored_df(hour, day_name)

    # 🔴 Firefighter
    firefighter = scored.iloc[0].to_dict() if len(scored) > 0 else None

    # 🟡 Preventative — Medium/Low/Very Low, trend=increasing
    prev = scored[
        scored["priority_tier"].isin(["Medium","Low","Very Low"]) &
        (scored["trend"] == "increasing")
    ]
    preventative = prev.iloc[0].to_dict() if len(prev) > 0 else None

    # 🔵 Wildcard — highest surge_pct, wildcard_eligible=1
    wild = scored[scored["wildcard_eligible"] == 1].sort_values("surge_pct", ascending=False)
    wildcard = wild.iloc[0].to_dict() if len(wild) > 0 else None

    return {"firefighter": firefighter, "preventative": preventative, "wildcard": wildcard}

def get_full_queue(hour, day_name, top_n=50):
    scored = get_scored_df(hour, day_name).head(top_n)
    return scored[[
        "zone_label","priority_tier","adj_score","ensemble_score",
        "peak_hour","peak_day","trend","surge_pct",
        "top_violations","top_vehicles","violation_count","lat","lon"
    ]].rename(columns={
        "zone_label"     : "Zone",
        "priority_tier"  : "Tier",
        "adj_score"      : "Adj Score",
        "ensemble_score" : "Base Score",
        "peak_hour"      : "Peak Hr",
        "peak_day"       : "Peak Day",
        "trend"          : "Trend",
        "surge_pct"      : "Surge %",
        "violation_count": "Total Violations",
    })


# ══════════════════════════════════════════════════════════════════════════
# GENAI HELPERS
# ══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """\
You are ParkSentinel, an AI assistant for Bengaluru Traffic Police.
Help traffic inspectors understand parking violations at ALL priority levels.

Rules:
- Always cite specific numbers from the data. Never invent figures.
- Cover ALL tiers — Critical, High, Medium, Low, and Very Low — when relevant.
- For low/medium zones trending upward, explain why they deserve early attention.
- Be operational: tell the inspector WHAT to do, WHERE, and WHEN.
- Keep answers under 250 words unless a detailed breakdown is asked for.
- Plain English only — inspectors are not data scientists.
"""

def build_context(hour=None, day_name=None):
    lines = ["=== BENGALURU PARKING VIOLATION INTELLIGENCE — FULL CITY ===\n"]

    # Current missions
    if hour is not None and day_name is not None:
        m = get_stratified_missions(hour, day_name)
        lines.append(f"== CURRENT MISSIONS ({hour:02d}:00 on {day_name}) ==")
        ff = m["firefighter"]
        pv = m["preventative"]
        wc = m["wildcard"]
        if ff:
            lines.append(f"🔴 FIREFIGHTER: {ff['zone_label']} "
                         f"(Adj Score:{ff['adj_score']:.0f}, Tier:{ff['priority_tier']})")
        if pv:
            lines.append(f"🟡 PREVENTATIVE: {pv['zone_label']} "
                         f"(Trend:{pv['trend']}, Surge:{pv.get('surge_pct',0):+.0f}%)")
        else:
            lines.append("🟡 PREVENTATIVE: None — all Medium/Low zones are stable or decreasing")
        if wc:
            lines.append(f"🔵 WILDCARD: {wc['zone_label']} "
                         f"(Surge:{wc.get('surge_pct',0):+.0f}%, "
                         f"Recent:{int(wc.get('recent_count',0))} violations)")
        lines.append("")

    tier_configs = [
        ("Critical",  20, "Immediate enforcement"),
        ("High",      15, "Active patrol rotation"),
        ("Medium",    10, "Preventative patrol target"),
        ("Low",        8, "Monitor — watch for trends"),
        ("Very Low",   5, "Early-warning watchlist"),
    ]
    for tier, limit, desc in tier_configs:
        t_zones = zone_df[zone_df["priority_tier"]==tier].nlargest(limit,"ensemble_score")
        if len(t_zones) == 0: continue
        total_in_tier = (zone_df["priority_tier"]==tier).sum()
        lines.append(f"\n== {tier.upper()} ({total_in_tier} zones total, top {len(t_zones)} shown) — {desc} ==")
        for _, r in t_zones.iterrows():
            tv   = r["top_violations"] if isinstance(r["top_violations"], list) else []
            tveh = r["top_vehicles"]   if isinstance(r["top_vehicles"],   list) else []
            lines.append(
                f"  • {r['zone_label']} | Score:{r['ensemble_score']:.0f} | "
                f"Violations:{r['violation_count']:,} | Peak:{r['peak_hour']}:00 {r['peak_day']} | "
                f"Trend:{r['trend']} | Surge:{r.get('surge_pct',0):+.0f}% | "
                f"Offenses:{', '.join(tv[:2]) if tv else 'N/A'} | "
                f"Vehicles:{', '.join(tveh[:1]) if tveh else 'N/A'}"
            )
    return "\n".join(lines)

def ask_gemini(question, use_pro=False, hour=None, day_name=None):
    if not gemini_flash:
        return "⚠️ Gemini not connected — add GOOGLE_API_KEY to .env"
    ctx    = build_context(hour=hour, day_name=day_name)
    prompt = f"{SYSTEM_PROMPT}\n\n{ctx}\n\nInspector: {question}\n\nParkSentinel:"
    model  = gemini_pro if use_pro else gemini_flash
    return model.generate_content(prompt).text


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚦 ParkSentinel")
    st.caption("Bengaluru Traffic Enforcement Intelligence")
    st.divider()

    page = st.radio("Navigate", [
        "🗺️  Overview Map",
        "🚨  Enforcement Queue",
        "🔍  Zone Deep Dive",
        "📅  Patrol Scheduler",
        "🤖  AI Assistant",
    ])
    st.divider()

    ist = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    st.markdown(f"**IST:** `{ist.strftime('%H:%M')}  {ist.strftime('%A')}`")
    st.markdown(f"**Date:** `{ist.strftime('%d %b %Y')}`")
    st.divider()

    total_v  = int(zone_df["violation_count"].sum())
    n_zones  = len(zone_df)
    n_crit   = int((zone_df["priority_tier"]=="Critical").sum())
    n_surge  = int(zone_df["wildcard_eligible"].sum())

    st.markdown(f"""
    <div class="kpi-box" style="margin-bottom:8px">
      <h3>{total_v:,}</h3><p>Total Violations (Jan–May)</p>
    </div>
    <div class="kpi-box" style="margin-bottom:8px">
      <h3>{n_zones}</h3><p>Total Zones Tracked</p>
    </div>
    <div class="kpi-box" style="margin-bottom:8px">
      <h3>{n_crit}</h3><p>Critical Zones</p>
    </div>
    <div class="kpi-box">
      <h3>{n_surge}</h3><p>Surging Zones (Wildcard eligible)</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW MAP
# ══════════════════════════════════════════════════════════════════════════
if page.startswith("🗺️"):
    st.title("🗺️ Bengaluru Parking Violation Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        min_score  = st.slider("Min Priority Score to show", 0, 80, 0, 5)
        map_style  = st.selectbox("Map Style",
            ["OpenStreetMap","CartoDB dark_matter","CartoDB positron"],
            index=0)
    with col2:
        show_heat  = st.checkbox("Show Violation Heatmap", value=True)
        show_hexes = st.checkbox("Show Priority Hex Zones", value=True)
    with col3:
        tier_filter = st.multiselect("Filter Tiers",
            TIER_ORDER, default=TIER_ORDER)

    m = folium.Map(location=[12.97,77.59], zoom_start=12, tiles=map_style)

    # Heatmap layer
    if show_heat:
        sample = vmap_df.sample(min(15_000, len(vmap_df)), random_state=42)
        heat_data = list(zip(
            sample["latitude"], sample["longitude"],
            sample["severity_score"].fillna(1)
        ))
        HeatMap(heat_data, radius=8, blur=10, max_zoom=14,
                gradient={"0.2":"blue","0.5":"lime","0.8":"orange","1.0":"red"},
                name="Violation Density").add_to(m)

    # Hex polygons
    drawn = 0
    if show_hexes:
        # Use enriched hex_df (has zone_label); fall back to hex_df
        map_hex = hex_df.copy()
        map_hex = map_hex[
            (map_hex["ensemble_score"] >= min_score) &
            (map_hex["final_priority_tier"].isin(tier_filter))
        ]

        for _, row in map_hex.iterrows():
            tier  = str(row.get("final_priority_tier","Very Low"))
            color = TIER_COLORS.get(tier,"#95a5a6")
            score = row["ensemble_score"]
            label = row.get("zone_label", row.get("police_station","—"))

            try:
                boundary   = h3.cell_to_boundary(row["h3_id"])
                poly_coords = [[lat,lon] for lat,lon in boundary]
            except Exception:
                continue

            popup_html = (
                f"<b>{label}</b><br>"
                f"<b>Tier:</b> {tier} | <b>Score:</b> {score:.1f}<br>"
                f"<b>Violations:</b> {int(row['violation_count']):,}<br>"
                f"<b>Severity avg:</b> {row['severity_mean']:.2f}<br>"
                f"<b>Near junction:</b> {row['near_junction_ratio']*100:.0f}%<br>"
                f"<b>Peak hour ratio:</b> {row['peak_hour_ratio']*100:.0f}%"
            )
            folium.Polygon(
                locations=poly_coords,
                color=color, fill=True,
                fill_color=color, fill_opacity=0.55, weight=1,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"{tier} — {label[:35]} ({score:.0f})"
            ).add_to(m)
            drawn += 1

    # Top 10 critical markers always shown
    top10 = hex_df.nlargest(10,"ensemble_score")
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        label = row.get("zone_label", row.get("police_station","—"))
        folium.Marker(
            location=[row["lat"],row["lon"]],
            tooltip=f"#{rank} {label[:30]} ({row['ensemble_score']:.0f})",
            popup=f"<b>#{rank}</b> {label}<br>Score:{row['ensemble_score']:.1f}",
            icon=folium.Icon(color="red", icon="exclamation-sign")
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=1200, height=580)
    st.caption(
        f"Showing {drawn} hex zones | Score ≥ {min_score} | "
        f"Tiers: {', '.join(tier_filter)} | "
        f"Click any hex for details. Red markers = top 10 critical zones."
    )

    # Tier legend
    legend_html = " &nbsp; ".join(
        f'<span style="color:{TIER_COLORS[t]}; font-size:1.1rem">■</span> {t}'
        for t in TIER_ORDER
    )
    st.markdown("**Legend:** " + legend_html, unsafe_allow_html=True)

    # Tier distribution mini chart
    with st.expander("📊 Zone Tier Distribution", expanded=False):
        tier_counts = zone_df["priority_tier"].value_counts().reindex(TIER_ORDER).fillna(0)
        fig = px.bar(
            x=tier_counts.index, y=tier_counts.values,
            color=tier_counts.index,
            color_discrete_map=TIER_COLORS,
            labels={"x":"Tier","y":"Number of Zones"},
            title=f"All {len(zone_df):,} Zones by Priority Tier"
        )
        fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font_color="white", showlegend=False,
            margin=dict(t=40,b=20,l=20,r=20)
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — ENFORCEMENT QUEUE
# ══════════════════════════════════════════════════════════════════════════
elif page.startswith("🚨"):
    st.title("🚨 Live Enforcement Queue")
    st.caption("Stratified patrol missions + full city-wide ranked queue")

    ist = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=5, minutes=30)))

    col1, col2, col3 = st.columns(3)
    with col1:
        query_hour = st.slider("Hour (IST)", 0, 23, ist.hour,
            help="Slide to see how patrol priorities shift across the day")
    with col2:
        query_day  = st.selectbox("Day of Week", DAY_ORDER, index=ist.weekday())
    with col3:
        top_n = st.slider("Ranked queue size", 10, 50, 20)

    # ── Stratified missions ───────────────────────────────────────────────
    st.markdown(f"### 🎯 Patrol Missions at **{query_hour:02d}:00** on **{query_day}**")
    missions = get_stratified_missions(query_hour, query_day)

    m_col1, m_col2, m_col3 = st.columns(3)

    def render_mission(col, icon, css_class, title, m, extra_field=None):
        with col:
            if m is None:
                st.markdown(f"""
                <div class="mission-card {css_class}">
                  <h4>{icon} {title}</h4>
                  <p>⚪ <i>No qualifying zone for this slot.</i><br>
                  All Medium/Low/Very Low zones are currently stable or decreasing.
                  </p>
                </div>""", unsafe_allow_html=True)
            else:
                tv   = m.get("top_violations",[]) if isinstance(m.get("top_violations",[]),list) else []
                tveh = m.get("top_vehicles",[])   if isinstance(m.get("top_vehicles",[]),list)   else []
                extra = ""
                if extra_field == "surge":
                    extra = f"<p>📈 Surge: <b>+{m.get('surge_pct',0):.0f}%</b> ({int(m.get('recent_count',0))} violations last 14 days)</p>"
                elif extra_field == "trend":
                    extra = f"<p>Trend: <b>{m.get('trend','—').upper()}</b> | Surge: {m.get('surge_pct',0):+.0f}%</p>"

                st.markdown(f"""
                <div class="mission-card {css_class}">
                  <h4>{icon} {title}</h4>
                  <p><b>{m['zone_label']}</b></p>
                  <p>Tier: <b>{m['priority_tier']}</b> &nbsp;|&nbsp;
                     Adj Score: <b>{m['adj_score']:.1f}</b> &nbsp;|&nbsp;
                     Base: {m['ensemble_score']:.1f}</p>
                  <p>Peak: {m['peak_hour']}:00 on {m['peak_day']}</p>
                  <p>Offenses: {', '.join(tv[:2]) if tv else '—'}</p>
                  <p>Vehicles: {', '.join(tveh[:1]) if tveh else '—'}</p>
                  {extra}
                </div>""", unsafe_allow_html=True)

    render_mission(m_col1, "🔴", "firefighter",  "#1 Firefighter",   missions["firefighter"])
    render_mission(m_col2, "🟡", "preventative", "Preventative",     missions["preventative"], "trend")
    render_mission(m_col3, "🔵", "wildcard",     "Wildcard Surge",   missions["wildcard"],     "surge")

    # ── Mini map of 3 missions ────────────────────────────────────────────
    st.markdown("#### Mission Locations")
    m2 = folium.Map(location=[12.97,77.59], zoom_start=12, tiles="OpenStreetMap")

    mission_map = {
        "🔴 Firefighter" : (missions["firefighter"],  "red",    "exclamation-sign"),
        "🟡 Preventative": (missions["preventative"], "orange", "eye-open"),
        "🔵 Wildcard"    : (missions["wildcard"],     "blue",   "fire"),
    }
    for label, (m_data, color, icon_name) in mission_map.items():
        if m_data is None: continue
        folium.Marker(
            location=[m_data["lat"], m_data["lon"]],
            tooltip=f"{label}: {m_data['zone_label'][:35]}",
            popup=f"<b>{label}</b><br>{m_data['zone_label']}<br>"
                  f"Score:{m_data['adj_score']:.1f}",
            icon=folium.Icon(color=color, icon=icon_name)
        ).add_to(m2)
    st_folium(m2, width=900, height=320)

    # ── Full ranked queue ─────────────────────────────────────────────────
    st.divider()
    st.markdown(f"### 📋 Full Ranked Queue — Top {top_n} Zones at {query_hour:02d}:00 on {query_day}")
    st.caption("Scores re-rank dynamically based on hour — slide to see changes")

    queue = get_full_queue(query_hour, query_day, top_n=top_n)

    # Format display columns
    disp = queue[["Zone","Tier","Adj Score","Base Score",
                  "Peak Hr","Peak Day","Trend","Surge %","Total Violations"]].copy()
    disp["Surge %"] = disp["Surge %"].apply(lambda x: f"{x:+.0f}%")
    disp["Peak Hr"] = disp["Peak Hr"].apply(lambda x: f"{int(x)}:00" if str(x).isdigit() else str(x))

    def color_tier(val):
        colors = {"Critical":"#e74c3c","High":"#e67e22","Medium":"#c8a800",
                  "Low":"#27ae60","Very Low":"#2980b9"}
        return f"color:{colors.get(val,'white')}; font-weight:bold"

    def color_trend(val):
        return ("color:#e74c3c" if val=="increasing" else
                "color:#2ecc71" if val=="decreasing" else "color:#aaa")

    styled = (disp.style
              .map(color_tier,   subset=["Tier"])
              .map(color_trend,  subset=["Trend"]))
    st.dataframe(styled, use_container_width=True, height=460)

    # ── Carriageway Impact Estimator ──────────────────────────────────────
    st.divider()
    st.markdown("#### 📏 Carriageway Impact Estimator")
    st.caption("Select any zone to estimate road space freed if violations are cleared.")

    all_zones_sorted = zone_df.sort_values("ensemble_score", ascending=False)["zone_label"].tolist()
    sel_zone = st.selectbox("Select zone", all_zones_sorted, key="impact_sel")
    sel_row  = zone_df[zone_df["zone_label"]==sel_zone]

    if len(sel_row) > 0:
        sel_row = sel_row.iloc[0]
        v_count  = int(sel_row["violation_count"])
        avg_len  = 4.5    # metres per vehicle (car average)
        lane_w   = 3.5    # metres
        freed_m  = v_count * avg_len
        freed_lanes = freed_m / lane_w
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Zone Tier",          sel_row["priority_tier"])
        e2.metric("Total Violations",   f"{v_count:,}")
        e3.metric("Carriageway Freed",  f"{freed_m:,.0f} m")
        e4.metric("Lane-metres Freed",  f"{freed_lanes:,.0f}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — ZONE DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════
elif page.startswith("🔍"):
    st.title("🔍 Zone Deep Dive")
    st.caption(f"Explore any of {len(zone_df):,} zones across all priority tiers")

    # Two-step selector: Tier → Zone
    col_t, col_z = st.columns([1,3])
    with col_t:
        sel_tier = st.selectbox("Filter by Tier", ["All"] + TIER_ORDER)
    with col_z:
        if sel_tier == "All":
            filtered = zone_df.sort_values("ensemble_score", ascending=False)
        else:
            filtered = zone_df[zone_df["priority_tier"]==sel_tier].sort_values(
                "ensemble_score", ascending=False)
        sel_zone = st.selectbox(
            f"Select Zone ({len(filtered)} in selection)",
            filtered["zone_label"].tolist()
        )

    row = zone_df[zone_df["zone_label"]==sel_zone].iloc[0]

    # KPI row
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Priority Score",    f"{row['ensemble_score']:.1f}")
    k2.metric("Tier",              row["priority_tier"])
    k3.metric("Total Violations",  f"{row['violation_count']:,}")
    k4.metric("Peak Hour",         f"{row['peak_hour']}:00 IST")
    k5.metric("Peak Day",          row["peak_day"])
    k6.metric("Trend",             row["trend"].upper())

    st.markdown(
        f"**Morning patrol:** `{row['best_morning_start']}:00–{row['best_morning_start']+2}:00` &nbsp;|&nbsp; "
        f"**Evening patrol:** `{row['best_evening_start']}:00–{row['best_evening_start']+2}:00` &nbsp;|&nbsp; "
        f"**Weekend share:** `{row['weekend_ratio']*100:.0f}%` &nbsp;|&nbsp; "
        f"**Near junction:** `{row['near_junction_ratio']*100:.0f}%` &nbsp;|&nbsp; "
        f"**Avg severity:** `{row['severity_mean']:.2f}/5` &nbsp;|&nbsp; "
        f"**Surge (14d):** `{row.get('surge_pct',0):+.0f}%`"
    )
    if row.get("wildcard_eligible",0):
        st.warning("🔵 This zone is **Wildcard eligible** — surging violations, watch closely.")

    st.divider()
    col_l, col_r = st.columns(2)

    # Hourly Plotly bar
    with col_l:
        hc     = row["hourly_counts"]
        hours  = list(range(24))
        counts = [hc.get(h, hc.get(str(h), 0)) for h in hours]
        colors = ["#e74c3c" if h in list(range(7,11))+list(range(17,22))
                  else "#3498db" for h in hours]
        fig = go.Figure(go.Bar(
            x=hours, y=counts, marker_color=colors,
            hovertemplate="Hour %{x}:00 — %{y} violations<extra></extra>"
        ))
        fig.update_layout(
            title="Hourly Violation Profile  (Red = Peak Hours)",
            xaxis_title="Hour (IST)", yaxis_title="Violations",
            paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
            font_color="white",
            xaxis=dict(tickvals=list(range(0,24,2))),
            margin=dict(t=40,b=30,l=40,r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Daily Plotly bar
    with col_r:
        dc       = row["daily_counts"]
        d_counts = [dc.get(d, 0) for d in DAY_ORDER]
        d_colors = ["#e74c3c" if d in ["Saturday","Sunday"] else "#2ecc71"
                    for d in DAY_ORDER]
        fig2 = go.Figure(go.Bar(
            x=[d[:3] for d in DAY_ORDER], y=d_counts,
            marker_color=d_colors,
            hovertemplate="%{x} — %{y} violations<extra></extra>"
        ))
        fig2.update_layout(
            title="Day-of-Week Profile  (Red = Weekend)",
            xaxis_title="Day", yaxis_title="Violations",
            paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
            font_color="white",
            margin=dict(t=40,b=30,l=40,r=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Violation + vehicle breakdown
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Top Violation Types**")
        tv = row["top_violations"] if isinstance(row["top_violations"],list) else []
        for v in tv: st.markdown(f"- {v}")
        if not tv: st.caption("No data")
    with col4:
        st.markdown("**Top Vehicle Types**")
        tveh = row["top_vehicles"] if isinstance(row["top_vehicles"],list) else []
        for v in tveh: st.markdown(f"- {v}")
        if not tveh: st.caption("No data")

    # Surge trend context
    st.divider()
    s1, s2, s3 = st.columns(3)
    s1.metric("Recent 14d violations", int(row.get("recent_count",0)))
    s2.metric("Prior 14d violations",  int(row.get("prior_count",0)))
    s3.metric("Surge",                 f"{row.get('surge_pct',0):+.0f}%",
              delta_color="inverse" if row.get("surge_pct",0) > 0 else "normal")

    # Zone location mini map
    st.divider()
    st.markdown("**Zone Location**")
    m3 = folium.Map(location=[row["lat"],row["lon"]], zoom_start=15,
                    tiles="OpenStreetMap")
    try:
        boundary = h3.cell_to_boundary(row["h3_id"])
        folium.Polygon(
            locations=[[lat,lon] for lat,lon in boundary],
            color=TIER_COLORS.get(row["priority_tier"],"#aaa"),
            fill=True, fill_opacity=0.4, weight=2
        ).add_to(m3)
    except Exception:
        pass
    folium.Marker(
        location=[row["lat"],row["lon"]],
        tooltip=sel_zone,
        icon=folium.Icon(color="red", icon="exclamation-sign")
    ).add_to(m3)
    st_folium(m3, width=900, height=320)

    # Compare with similar-tier zones
    with st.expander(f"Compare with other {row['priority_tier']} zones", expanded=False):
        peers = zone_df[
            (zone_df["priority_tier"]==row["priority_tier"]) &
            (zone_df["zone_label"]!=sel_zone)
        ].nlargest(10,"ensemble_score")[[
            "zone_label","ensemble_score","violation_count",
            "peak_hour","peak_day","trend","surge_pct"
        ]].rename(columns={
            "zone_label":"Zone","ensemble_score":"Score",
            "violation_count":"Violations","peak_hour":"Peak Hr",
            "peak_day":"Peak Day","surge_pct":"Surge %"
        })
        peers["Surge %"] = peers["Surge %"].apply(lambda x: f"{x:+.0f}%")
        st.dataframe(peers, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — PATROL SCHEDULER
# ══════════════════════════════════════════════════════════════════════════
elif page.startswith("📅"):
    st.title("📅 Weekly Stratified Patrol Scheduler")
    st.caption("Three mission types per time slot — Firefighter, Preventative, Wildcard")

    # Mission type selector
    mission_type = st.radio(
        "View missions of type:",
        ["🔴 Firefighter","🟡 Preventative","🔵 Wildcard"],
        horizontal=True
    )
    mtype_key = {"🔴 Firefighter":"firefighter",
                 "🟡 Preventative":"preventative",
                 "🔵 Wildcard":"wildcard"}[mission_type]

    # Filter calendar
    cal_filtered = calendar_df[calendar_df["mission"]==mtype_key]
    slot_labels  = [s[1] for s in SLOTS]

    # Heatmap of scores
    pivot_score = cal_filtered.pivot_table(
        index="day", columns="time_slot", values="adj_score", aggfunc="first"
    ).reindex(DAY_ORDER).fillna(0)

    fig_heat = px.imshow(
        pivot_score,
        color_continuous_scale="YlOrRd",
        labels={"color":"Priority Score"},
        title=f"{mission_type} — Priority Score per Slot",
        aspect="auto",
        text_auto=".0f"
    )
    fig_heat.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#1e2130", font_color="white",
        margin=dict(t=50,b=20,l=20,r=20)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Zone name heatmap
    pivot_names = cal_filtered.pivot_table(
        index="day", columns="time_slot", values="zone_label",
        aggfunc="first"
    ).reindex(DAY_ORDER).fillna("—")
    # Truncate names
    pivot_names_short = pivot_names.map(
        lambda x: x[:18] if isinstance(x,str) else "—"
    )
    # Show as a styled dataframe (no matplotlib)
    st.markdown(f"**Zone assigned per slot — {mission_type}**")
    st.dataframe(pivot_names_short, use_container_width=True)

    # Drill-down: specific slot — top 15 zones
    st.divider()
    st.markdown("#### Slot Drill-Down — Full Zone Rankings")
    st.caption("Shows all zones for the selected slot, ranked by time-adjusted score")

    dc1, dc2 = st.columns(2)
    with dc1: drill_day  = st.selectbox("Day", DAY_ORDER)
    with dc2: drill_slot = st.selectbox("Time Slot", slot_labels)

    slot_hour = dict(SLOTS).get(
        next((h for h,l in SLOTS if l==drill_slot), 6), 6)

    drill_queue = get_full_queue(slot_hour, drill_day, top_n=50)
    drill_disp  = drill_queue[["Zone","Tier","Adj Score","Base Score",
                                "Trend","Surge %","Total Violations"]].copy()
    drill_disp["Surge %"] = drill_disp["Surge %"].apply(lambda x: f"{x:+.0f}%")

    def color_tier_d(val):
        c = {"Critical":"#e74c3c","High":"#e67e22","Medium":"#c8a800",
             "Low":"#27ae60","Very Low":"#2980b9"}
        return f"color:{c.get(val,'white')}; font-weight:bold"

    def color_trend_d(val):
        return ("color:#e74c3c" if val=="increasing" else
                "color:#2ecc71" if val=="decreasing" else "color:#aaa")

    styled_d = (drill_disp.style
                .map(color_tier_d,  subset=["Tier"])
                .map(color_trend_d, subset=["Trend"]))

    st.dataframe(styled_d, use_container_width=True, height=520)
    st.caption(f"Showing top 50 zones for {drill_day} {drill_slot}")

    # Download schedule CSV
    readable = cal_filtered[cal_filtered["mission"]==mtype_key].pivot_table(
        index="day", columns="time_slot", values="zone_label", aggfunc="first"
    ).reindex(DAY_ORDER).fillna("—")
    st.download_button(
        f"⬇️ Download {mission_type} Schedule CSV",
        readable.to_csv(),
        f"patrol_{mtype_key}_schedule.csv",
        "text/csv"
    )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════
elif page.startswith("🤖"):
    st.title("🤖 ParkSentinel AI Assistant")

    if not gemini_flash:
        st.error("Gemini not connected. Add GOOGLE_API_KEY to your .env file and restart.")
        st.stop()

    ist = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=5, minutes=30)))

    # Controls row
    ctl1, ctl2, ctl3 = st.columns([2,2,1])
    with ctl1:
        use_pro = st.toggle("Use Gemini 2.5 Pro (slower, more thorough)", value=False)
    with ctl2:
        inject_time = st.toggle("Inject current patrol missions into context", value=True)
    with ctl3:
        if st.button("🗑️ Clear Chat"):
            st.session_state["messages"] = []
            st.rerun()

    model_name = "gemini-2.5-pro" if use_pro else "gemini-2.5-flash"
    st.caption(
        f"Model: `{model_name}` | Context: ALL tiers (Critical→Very Low) + "
        f"{'current missions' if inject_time else 'static context'} | "
        f"IST: {ist.strftime('%H:%M on %A')}"
    )

    # Sidebar quick questions
    col_chat, col_quick = st.columns([3,1])

    with col_quick:
        st.markdown("**Quick Questions**")
        quick_qs = [
            "Where should I patrol right now?",
            "Top 3 critical zones and why?",
            "Which low/medium zones are trending up?",
            "Which vehicles cause worst congestion?",
            "Best weekend patrol strategy?",
            "Which zones are surging this week?",
            "Compare critical vs medium zone patterns.",
            "Give me a full enforcement briefing.",
        ]
        for q in quick_qs:
            if st.button(q, key=f"qq_{q}", use_container_width=True):
                st.session_state["prefill"] = q

    with col_chat:
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Render history
        for msg in st.session_state["messages"]:
            avatar = "👮" if msg["role"]=="user" else "🚦"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        # Input
        prefill    = st.session_state.pop("prefill","")
        user_input = st.chat_input(
            "Ask about any zone, tier, trend, or patrol strategy…"
        ) or prefill

        if user_input:
            with st.chat_message("user", avatar="👮"):
                st.markdown(user_input)
            st.session_state["messages"].append(
                {"role":"user","content":user_input})

            h_ctx = ist.hour      if inject_time else None
            d_ctx = ist.strftime("%A") if inject_time else None

            with st.chat_message("assistant", avatar="🚦"):
                with st.spinner("Analysing full city data…"):
                    response = ask_gemini(user_input, use_pro=use_pro,
                                          hour=h_ctx, day_name=d_ctx)
                st.markdown(response)

            st.session_state["messages"].append(
                {"role":"assistant","content":response})
