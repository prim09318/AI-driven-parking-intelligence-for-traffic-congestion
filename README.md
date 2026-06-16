# 🚦 ParkSentinel — AI-Driven Parking Violation Intelligence
### Gridlock Hackathon 2.0 | Flipkart × Bengaluru Traffic Police

> *Transforming reactive parking enforcement into proactive, data-driven patrolling.*

---

## Problem
On-street illegal parking near commercial zones, metro stations, and junctions
chokes Bengaluru's carriageways daily. Enforcement is patrol-based and reactive —
officers have no way to know **where** violations are clustering or **when** to
patrol for maximum impact.

## Solution
**ParkSentinel** is an AI-powered enforcement intelligence system that:
- Detects parking violation hotspots using spatial clustering (H3 + DBSCAN)
- Scores each zone by congestion-risk impact (severity × vehicle size × road type × junction proximity)
- Forecasts **when** each zone will peak using temporal models
- Provides a natural-language AI assistant for non-technical inspectors (Gemini Flash)

---

## Project Structure
```
parkssentinel/
├── data/
│   ├── raw/                    # Original dataset (not committed)
│   └── processed/              # Cleaned parquet files
├── notebooks/
│   ├── day1_pipeline_eda.ipynb     # Data pipeline + EDA
│   ├── day2_hotspot_model.ipynb    # H3 binning + DBSCAN + priority score
│   ├── day3_temporal.ipynb         # Temporal forecasting per zone
│   └── day4_explainability.ipynb   # SHAP + GenAI explainer
├── src/
│   ├── config.py               # Paths, constants, severity weights
│   ├── pipeline.py             # Data loading + cleaning functions
│   ├── features.py             # Feature engineering
│   ├── model.py                # Priority scoring model
│   ├── temporal.py             # Forecasting functions
│   └── genai_assistant.py      # Gemini Flash integration
├── app/
│   └── streamlit_app.py        # Main dashboard
├── models/                     # Saved model artifacts
├── .env.example                # Environment variable template
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and set up environment
```bash
git clone <your-repo-url>
cd parksentinel
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Add data
```bash
mkdir -p data/raw
# Copy the dataset CSV into data/raw/
```

### 4. Run notebooks in order
```
notebooks/day1_pipeline_eda.ipynb       → cleans data, saves parquet
notebooks/day2_hotspot_model.ipynb      → builds priority scores
notebooks/day3_temporal.ipynb           → builds patrol schedules
```

### 5. Launch dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## Key Features

| Feature | Description |
|---|---|
| **Violation Heatmap** | H3 hexagonal choropleth of Bengaluru, coloured by priority score |
| **Enforcement Queue** | Ranked list of zones to patrol right now, updated by hour |
| **Zone Deep Dive** | Per-zone breakdown: vehicle types, peak hours, violation mix |
| **Patrol Scheduler** | Hour × Day enforcement calendar per zone |
| **AI Assistant** | Ask plain-English questions — Gemini Flash answers with data context |
| **Impact Estimator** | Estimated carriageway freed if violations in a zone are cleared |

---

## Data
- Source: Bengaluru Traffic Police via HackerEarth (Jan–May, ~3 lakh records)
- Fields: GPS coordinates, vehicle type, violation codes, timestamp, junction name, validation status
- **Note:** Raw data is not committed to this repository per competition guidelines.

---

## Technical Stack
- **Pipeline:** pandas, pyarrow
- **Geospatial:** H3, GeoPandas, Folium
- **ML:** scikit-learn (DBSCAN), XGBoost, SHAP
- **Forecasting:** Prophet
- **GenAI:** Google Gemini Flash via LangChain
- **Dashboard:** Streamlit + streamlit-folium

---

## Limitations & Honesty
- Dataset contains **violation records**, not traffic flow data.
  Congestion impact is a weighted **proxy score**, not a direct measurement.
- Temporal forecasting relies on historical patterns — unexpected events
  (rallies, festivals) are not accounted for.
- Future work: integrate CCTV feeds, OpenStreetMap road capacity, and
  real-time Waze/Google Maps traffic data.

---

## Author
Gridlock Hackathon 2.0 — Round 2 Submission
