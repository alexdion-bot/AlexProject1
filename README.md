# Sovereign Risk Dashboard

Interactive dashboard for monitoring and comparing sovereign risk across 15 countries, combining market, macroeconomic, and governance indicators into a composite risk index.

## Features

- **Composite Risk Index** — Weighted score (0–100) combining 6 sub-indices
- **15 Countries** — Developed, emerging, and frontier markets (US, Germany, Japan, UK, France, Brazil, Mexico, South Africa, Turkey, Argentina, India, China, Russia, Nigeria, Egypt)
- **Interactive Visualizations** — Radar charts, heatmaps, time series, scatter plots
- **Real-time Data** — World Bank API integration with offline fallback
- **Export** — Download risk data as CSV

## Risk Indicators

| Indicator | Source | Weight |
|-----------|--------|--------|
| CDS Spreads | Simulated (realistic) | 25% |
| Debt-to-GDP | World Bank API | 20% |
| Political Risk | World Bank Governance Indicators | 15% |
| Inflation | World Bank API | 15% |
| Current Account Balance | World Bank API | 15% |
| FX Reserves / GDP | World Bank API | 10% |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data (optional, for offline use)
python generate_sample_data.py

# Launch the dashboard
streamlit run app.py
```

## Project Structure

```
.
├── app.py                    # Streamlit dashboard (main entry point)
├── src/
│   ├── data_collection.py    # World Bank API client + synthetic CDS data
│   └── risk_index.py         # Normalization + composite index computation
├── data/
│   └── cached_data.json      # Cached sample data for offline demo
├── generate_sample_data.py   # Script to regenerate sample data
└── requirements.txt
```

## Methodology

Each indicator is **min-max normalized** to a 0–100 scale per year across all countries. The composite score is a weighted average of all available sub-indices. Higher scores indicate higher risk.

**Risk Categories:**
- Very High (75–100) | High (55–75) | Moderate (35–55) | Low (15–35) | Very Low (0–15)

## Stack

- **pandas** — Data manipulation
- **requests** — API calls (World Bank)
- **Streamlit** — Web dashboard framework
- **Plotly** — Interactive visualizations

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the main file
4. Deploy — free hosting with a shareable URL for your CV
