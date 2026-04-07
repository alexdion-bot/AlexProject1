"""
Sovereign Risk Dashboard — Interactive Streamlit Application

An analytical tool for comparing sovereign risk across 15 countries,
using CDS spreads, debt/GDP, FX reserves, political risk, inflation,
and current account data.

Author: Alex
Stack: Streamlit, Plotly, Pandas, World Bank API
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import os

from src.data_collection import COUNTRIES, build_complete_dataset
from src.risk_index import (
    build_risk_index,
    get_risk_category,
    WEIGHTS,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sovereign Risk Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .risk-very-high { color: #d32f2f; font-weight: bold; }
    .risk-high { color: #f57c00; font-weight: bold; }
    .risk-moderate { color: #fbc02d; font-weight: bold; }
    .risk-low { color: #388e3c; font-weight: bold; }
    .risk-very-low { color: #1b5e20; font-weight: bold; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 4px solid #1976d2;
    }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px 15px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
CACHE_FILE = os.path.join(os.path.dirname(__file__), "data", "cached_data.json")


@st.cache_data(ttl=3600, show_spinner="Loading data from APIs...")
def load_data():
    """Load data with caching. Try APIs first, fall back to cached file."""
    raw = build_complete_dataset()

    # Check if we got meaningful data
    has_data = any(not df.empty for df in raw.values())

    if not has_data and os.path.exists(CACHE_FILE):
        st.info("Using cached sample data (API unavailable)")
        with open(CACHE_FILE) as f:
            cached = json.load(f)
        raw = {k: pd.DataFrame(v) for k, v in cached.items()}

    return raw


@st.cache_data(ttl=3600)
def compute_risk(raw_json: str):
    """Compute risk index from serialized raw data."""
    raw = {k: pd.DataFrame(v) for k, v in json.loads(raw_json).items()}
    return build_risk_index(raw)


def save_cache(raw_data: dict):
    """Save fetched data to local cache for offline use."""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    serializable = {k: df.to_dict(orient="records") for k, df in raw_data.items() if not df.empty}
    with open(CACHE_FILE, "w") as f:
        json.dump(serializable, f)


# ---------------------------------------------------------------------------
# Color scales
# ---------------------------------------------------------------------------
RISK_COLORS = {
    "Very High": "#d32f2f",
    "High": "#f57c00",
    "Moderate": "#fbc02d",
    "Low": "#388e3c",
    "Very Low": "#1b5e20",
}

REGION_MAP = {
    "US": "North America", "DE": "Europe", "JP": "Asia-Pacific",
    "GB": "Europe", "FR": "Europe", "BR": "Latin America",
    "MX": "Latin America", "ZA": "Africa", "TR": "Europe",
    "AR": "Latin America", "IN": "Asia-Pacific", "CN": "Asia-Pacific",
    "RU": "Europe", "NG": "Africa", "EG": "Africa",
}


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.title("Sovereign Risk Dashboard")
    st.markdown(
        "Composite risk index for **15 countries** based on CDS spreads, "
        "debt/GDP, FX reserves, governance indicators, inflation, and "
        "current account balance."
    )

    # Load and process data
    raw_data = load_data()

    if not raw_data:
        st.error("No data available. Please check your internet connection and reload.")
        return

    # Save cache for offline use
    save_cache(raw_data)

    # Serialize for caching
    raw_json = json.dumps(
        {k: df.to_dict(orient="records") for k, df in raw_data.items() if isinstance(df, pd.DataFrame) and not df.empty}
    )
    risk_df = compute_risk(raw_json)

    if risk_df.empty:
        st.error("Could not compute risk index. Insufficient data.")
        return

    # -------------------------------------------------------------------
    # Sidebar filters
    # -------------------------------------------------------------------
    st.sidebar.header("Filters")

    available_years = sorted(risk_df["year"].dropna().unique().astype(int))
    selected_year = st.sidebar.select_slider(
        "Select Year",
        options=available_years,
        value=max(available_years),
    )

    available_countries = sorted(risk_df["country"].dropna().unique())
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        options=available_countries,
        default=available_countries,
    )

    sub_index_options = [c for c in risk_df.columns if c not in
                         ["country_code", "country", "year", "composite_risk"]]
    selected_indicators = st.sidebar.multiselect(
        "Sub-indices to display",
        options=sub_index_options,
        default=sub_index_options,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Methodology")
    st.sidebar.markdown(
        "Each indicator is **min-max normalized** (0–100) per year across "
        "all countries. The composite score is a **weighted average**:"
    )
    for name, weight in WEIGHTS.items():
        st.sidebar.markdown(f"- **{name.replace('_', ' ').title()}**: {weight:.0%}")

    # -------------------------------------------------------------------
    # Filter data
    # -------------------------------------------------------------------
    year_data = risk_df[
        (risk_df["year"] == selected_year) &
        (risk_df["country"].isin(selected_countries))
    ].copy()
    year_data["risk_category"] = year_data["composite_risk"].apply(get_risk_category)
    year_data["region"] = year_data["country_code"].map(REGION_MAP)

    ts_data = risk_df[risk_df["country"].isin(selected_countries)].copy()

    # -------------------------------------------------------------------
    # TAB LAYOUT
    # -------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", "Country Comparison", "Time Series", "Data Explorer"
    ])

    # ===================================================================
    # TAB 1: OVERVIEW
    # ===================================================================
    with tab1:
        st.subheader(f"Risk Overview — {selected_year}")

        # Top metrics
        if not year_data.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                riskiest = year_data.loc[year_data["composite_risk"].idxmax()]
                st.metric("Highest Risk", riskiest["country"],
                          f"{riskiest['composite_risk']:.1f}/100")
            with col2:
                safest = year_data.loc[year_data["composite_risk"].idxmin()]
                st.metric("Lowest Risk", safest["country"],
                          f"{safest['composite_risk']:.1f}/100")
            with col3:
                avg = year_data["composite_risk"].mean()
                st.metric("Average Risk Score", f"{avg:.1f}/100")
            with col4:
                spread = year_data["composite_risk"].max() - year_data["composite_risk"].min()
                st.metric("Risk Spread", f"{spread:.1f} pts")

        # Choropleth-style bar chart (sorted)
        st.markdown("#### Composite Risk Ranking")
        sorted_data = year_data.sort_values("composite_risk", ascending=True)

        fig_bar = px.bar(
            sorted_data,
            x="composite_risk",
            y="country",
            orientation="h",
            color="composite_risk",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 100],
            labels={"composite_risk": "Risk Score", "country": ""},
            hover_data=["risk_category", "region"],
        )
        fig_bar.update_layout(
            height=500,
            yaxis={"categoryorder": "total ascending"},
            coloraxis_colorbar={"title": "Risk"},
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Risk distribution by region
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Risk by Region")
            if "region" in year_data.columns:
                fig_box = px.box(
                    year_data, x="region", y="composite_risk",
                    color="region", points="all",
                    labels={"composite_risk": "Risk Score", "region": "Region"},
                    hover_data=["country"],
                )
                fig_box.update_layout(
                    height=400, showlegend=False,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_box, use_container_width=True)

        with col_right:
            st.markdown("#### Risk Category Distribution")
            cat_counts = year_data["risk_category"].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            fig_pie = px.pie(
                cat_counts, values="count", names="category",
                color="category",
                color_discrete_map=RISK_COLORS,
            )
            fig_pie.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ===================================================================
    # TAB 2: COUNTRY COMPARISON
    # ===================================================================
    with tab2:
        st.subheader(f"Country Comparison — {selected_year}")

        # Radar chart for selected countries
        compare_countries = st.multiselect(
            "Select countries to compare (max 5 for readability)",
            options=sorted(year_data["country"].unique()),
            default=sorted(year_data["country"].unique())[:4],
            max_selections=5,
            key="radar_countries",
        )

        if compare_countries and selected_indicators:
            fig_radar = go.Figure()
            for country in compare_countries:
                row = year_data[year_data["country"] == country]
                if row.empty:
                    continue
                row = row.iloc[0]
                values = [row.get(ind, 0) for ind in selected_indicators]
                values.append(values[0])  # close the polygon

                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=[i.replace("_", " ").title() for i in selected_indicators] +
                          [selected_indicators[0].replace("_", " ").title()],
                    name=country,
                    fill="toself",
                    opacity=0.6,
                ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                height=550,
                margin=dict(l=60, r=60, t=40, b=40),
                title="Risk Profile Comparison (0 = Low Risk, 100 = High Risk)",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Heatmap of all sub-indices
        st.markdown("#### Sub-Index Heatmap")
        if selected_indicators:
            heatmap_data = year_data.set_index("country")[selected_indicators]
            heatmap_data = heatmap_data.sort_values(
                selected_indicators[0] if selected_indicators else "cds_spreads",
                ascending=False,
            )

            fig_heat = px.imshow(
                heatmap_data.values,
                x=[i.replace("_", " ").title() for i in selected_indicators],
                y=heatmap_data.index.tolist(),
                color_continuous_scale="RdYlGn_r",
                zmin=0, zmax=100,
                labels=dict(color="Risk Score"),
                aspect="auto",
            )
            fig_heat.update_layout(
                height=500,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # ===================================================================
    # TAB 3: TIME SERIES
    # ===================================================================
    with tab3:
        st.subheader("Risk Evolution Over Time")

        ts_metric = st.selectbox(
            "Select metric",
            options=["composite_risk"] + selected_indicators,
            format_func=lambda x: x.replace("_", " ").title(),
        )

        ts_countries = st.multiselect(
            "Select countries",
            options=sorted(ts_data["country"].unique()),
            default=sorted(ts_data["country"].unique())[:6],
            key="ts_countries",
        )

        if ts_countries and ts_metric:
            plot_data = ts_data[ts_data["country"].isin(ts_countries)].copy()
            plot_data = plot_data.dropna(subset=[ts_metric])

            fig_ts = px.line(
                plot_data,
                x="year", y=ts_metric,
                color="country",
                markers=True,
                labels={
                    ts_metric: ts_metric.replace("_", " ").title() + " Score",
                    "year": "Year",
                    "country": "Country",
                },
            )
            fig_ts.update_layout(
                height=500,
                hovermode="x unified",
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        # Year-over-year change
        st.markdown("#### Year-over-Year Change in Composite Risk")
        if len(available_years) >= 2:
            prev_year = available_years[-2] if selected_year == available_years[-1] else selected_year - 1
            if prev_year in available_years:
                current = risk_df[risk_df["year"] == selected_year][["country", "composite_risk"]].copy()
                previous = risk_df[risk_df["year"] == prev_year][["country", "composite_risk"]].copy()
                current = current.rename(columns={"composite_risk": "current"})
                previous = previous.rename(columns={"composite_risk": "previous"})
                yoy = current.merge(previous, on="country")
                yoy["change"] = yoy["current"] - yoy["previous"]
                yoy = yoy[yoy["country"].isin(selected_countries)]
                yoy = yoy.sort_values("change", ascending=True)

                fig_yoy = px.bar(
                    yoy, x="change", y="country",
                    orientation="h",
                    color="change",
                    color_continuous_scale="RdYlGn_r",
                    color_continuous_midpoint=0,
                    labels={"change": f"Change ({prev_year}→{selected_year})", "country": ""},
                )
                fig_yoy.update_layout(
                    height=450,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_yoy, use_container_width=True)

    # ===================================================================
    # TAB 4: DATA EXPLORER
    # ===================================================================
    with tab4:
        st.subheader("Raw Data Explorer")

        st.markdown("#### Risk Scores Table")
        display_cols = ["country_code", "country", "year", "composite_risk", "risk_category"] + \
                       [c for c in selected_indicators if c in year_data.columns]

        year_data_display = year_data.copy()
        if "risk_category" not in year_data_display.columns:
            year_data_display["risk_category"] = year_data_display["composite_risk"].apply(get_risk_category)

        available_display_cols = [c for c in display_cols if c in year_data_display.columns]
        st.dataframe(
            year_data_display[available_display_cols]
            .sort_values("composite_risk", ascending=False)
            .reset_index(drop=True)
            .style.format({c: "{:.1f}" for c in available_display_cols if c not in ["country_code", "country", "year", "risk_category"]})
            .background_gradient(subset=["composite_risk"], cmap="RdYlGn_r", vmin=0, vmax=100),
            use_container_width=True,
            height=500,
        )

        # Scatter plot: any two indicators
        st.markdown("#### Scatter: Compare Two Indicators")
        if len(selected_indicators) >= 2:
            col_x, col_y = st.columns(2)
            with col_x:
                x_var = st.selectbox("X-axis", selected_indicators, index=0)
            with col_y:
                y_var = st.selectbox("Y-axis", selected_indicators,
                                     index=min(1, len(selected_indicators) - 1))

            scatter_data = year_data.dropna(subset=[x_var, y_var])
            fig_scatter = px.scatter(
                scatter_data,
                x=x_var, y=y_var,
                color="composite_risk",
                color_continuous_scale="RdYlGn_r",
                range_color=[0, 100],
                size="composite_risk",
                text="country_code",
                hover_data=["country", "composite_risk"],
                labels={
                    x_var: x_var.replace("_", " ").title(),
                    y_var: y_var.replace("_", " ").title(),
                },
            )
            fig_scatter.update_traces(textposition="top center")
            fig_scatter.update_layout(
                height=500,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Download button
        st.markdown("#### Export Data")
        csv = risk_df[risk_df["country"].isin(selected_countries)].to_csv(index=False)
        st.download_button(
            label="Download risk data as CSV",
            data=csv,
            file_name="sovereign_risk_data.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
