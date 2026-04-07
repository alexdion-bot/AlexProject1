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
import numpy as np
import json
import os

from src.data_collection import COUNTRIES, build_complete_dataset
from src.risk_index import (
    build_risk_index,
    get_risk_category,
    WEIGHTS,
)

# -- Page config --
st.set_page_config(
    page_title="Sovereign Risk Index",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Dark theme CSS (Harvard Atlas inspired) --
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background-color: #0a0e17;
        color: #c8cdd5;
        font-family: 'Inter', sans-serif;
    }

    header[data-testid="stHeader"] {
        background-color: #0a0e17;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f1520;
        border-right: 1px solid #1a2234;
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #8892a0;
        font-size: 0.85rem;
    }

    h1 {
        color: #e8ecf1 !important;
        font-weight: 600 !important;
        font-size: 1.6rem !important;
        letter-spacing: -0.02em;
    }

    h2, h3, .stTabs [data-baseweb="tab"] {
        color: #b0b8c4 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #0f1520;
        border-radius: 4px;
        padding: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 3px;
        font-size: 0.8rem !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1a2538 !important;
        color: #e8ecf1 !important;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 14px 18px;
    }

    div[data-testid="stMetric"] label {
        color: #6b7280 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e5e7eb !important;
        font-size: 1.1rem !important;
        font-weight: 600;
    }

    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #9ca3af !important;
        font-size: 0.8rem !important;
    }

    .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: #8892a0 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .stDataFrame {
        border: 1px solid #1e293b;
        border-radius: 4px;
    }

    .stDownloadButton button {
        background-color: #1a2538 !important;
        color: #c8cdd5 !important;
        border: 1px solid #2a3a52 !important;
        border-radius: 4px;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .stDownloadButton button:hover {
        background-color: #243352 !important;
        border-color: #3a5a82 !important;
    }

    .subtitle-text {
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: -10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -- Plotly dark template --
PLOT_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#0f1520",
        font=dict(family="Inter, sans-serif", color="#8892a0", size=12),
        title=dict(font=dict(color="#b0b8c4", size=14)),
        xaxis=dict(
            gridcolor="#1a2234", zerolinecolor="#1a2234",
            tickfont=dict(color="#6b7280"),
        ),
        yaxis=dict(
            gridcolor="#1a2234", zerolinecolor="#1a2234",
            tickfont=dict(color="#6b7280"),
        ),
        coloraxis=dict(
            colorbar=dict(
                tickfont=dict(color="#6b7280"),
                title=dict(font=dict(color="#8892a0")),
            )
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892a0", size=11),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )
)

# Muted but distinct color palette (atlas-style)
COUNTRY_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#5fa2ce", "#fc7d0b", "#d62728", "#1f9e89", "#8c564b",
]

RISK_SCALE = [
    [0.0, "#1b5e20"],
    [0.15, "#2e7d32"],
    [0.35, "#f9a825"],
    [0.55, "#f57f17"],
    [0.75, "#e65100"],
    [1.0, "#b71c1c"],
]


def apply_dark_layout(fig, height=480):
    fig.update_layout(
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#0f1520",
        font=dict(family="Inter, sans-serif", color="#8892a0", size=12),
        xaxis=dict(gridcolor="#1a2234", zerolinecolor="#1a2234"),
        yaxis=dict(gridcolor="#1a2234", zerolinecolor="#1a2234"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a0")),
        margin=dict(l=0, r=10, t=30, b=0),
        height=height,
    )
    return fig


# -- Data loading --
CACHE_FILE = os.path.join(os.path.dirname(__file__), "data", "cached_data.json")


@st.cache_data(ttl=3600, show_spinner="Fetching data...")
def load_data():
    raw = build_complete_dataset()
    has_data = any(not df.empty for df in raw.values())
    if not has_data and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cached = json.load(f)
        raw = {k: pd.DataFrame(v) for k, v in cached.items()}
    return raw


@st.cache_data(ttl=3600)
def compute_risk(raw_json: str):
    raw = {k: pd.DataFrame(v) for k, v in json.loads(raw_json).items()}
    return build_risk_index(raw)


def save_cache(raw_data: dict):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    serializable = {k: df.to_dict(orient="records") for k, df in raw_data.items() if not df.empty}
    with open(CACHE_FILE, "w") as f:
        json.dump(serializable, f)


REGION_MAP = {
    "US": "North America", "DE": "Europe", "JP": "Asia-Pacific",
    "GB": "Europe", "FR": "Europe", "BR": "Latin America",
    "MX": "Latin America", "ZA": "Sub-Saharan Africa", "TR": "Europe",
    "AR": "Latin America", "IN": "Asia-Pacific", "CN": "Asia-Pacific",
    "RU": "Europe", "NG": "Sub-Saharan Africa", "EG": "MENA",
}


# -- Main --
def main():
    st.title("Sovereign Risk Index")
    st.markdown(
        '<p class="subtitle-text">Composite risk scoring for 15 sovereigns '
        '// CDS spreads, fiscal metrics, governance, external balances</p>',
        unsafe_allow_html=True,
    )

    raw_data = load_data()
    if not raw_data:
        st.error("No data available.")
        return

    save_cache(raw_data)

    raw_json = json.dumps(
        {k: df.to_dict(orient="records") for k, df in raw_data.items()
         if isinstance(df, pd.DataFrame) and not df.empty}
    )
    risk_df = compute_risk(raw_json)
    if risk_df.empty:
        st.error("Insufficient data to compute risk index.")
        return

    # -- Sidebar --
    st.sidebar.markdown("### Filters")

    available_years = sorted(risk_df["year"].dropna().unique().astype(int))
    selected_year = st.sidebar.select_slider(
        "Year", options=available_years, value=max(available_years),
    )

    available_countries = sorted(risk_df["country"].dropna().unique())
    selected_countries = st.sidebar.multiselect(
        "Countries", options=available_countries, default=available_countries,
    )

    sub_index_options = [c for c in risk_df.columns
                         if c not in ["country_code", "country", "year", "composite_risk"]]
    selected_indicators = st.sidebar.multiselect(
        "Sub-indices", options=sub_index_options, default=sub_index_options,
    )

    st.sidebar.markdown("")
    st.sidebar.markdown("### Methodology")
    st.sidebar.markdown(
        "Min-max normalization (0-100) per year. "
        "Composite = weighted average:"
    )
    for name, weight in WEIGHTS.items():
        label = name.replace("_", " ").title()
        st.sidebar.markdown(f"**{label}** {weight:.0%}")

    # -- Filter --
    year_data = risk_df[
        (risk_df["year"] == selected_year) &
        (risk_df["country"].isin(selected_countries))
    ].copy()
    year_data["risk_category"] = year_data["composite_risk"].apply(get_risk_category)
    year_data["region"] = year_data["country_code"].map(REGION_MAP)

    ts_data = risk_df[risk_df["country"].isin(selected_countries)].copy()

    # -- Tabs --
    tab1, tab2, tab3, tab4 = st.tabs([
        "OVERVIEW", "COMPARISON", "TIME SERIES", "DATA"
    ])

    # =============== TAB 1: OVERVIEW ===============
    with tab1:
        if not year_data.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                r = year_data.loc[year_data["composite_risk"].idxmax()]
                st.metric("Highest Risk", r["country"], f"{r['composite_risk']:.1f}")
            with col2:
                s = year_data.loc[year_data["composite_risk"].idxmin()]
                st.metric("Lowest Risk", s["country"], f"{s['composite_risk']:.1f}")
            with col3:
                st.metric("Mean", f"{year_data['composite_risk'].mean():.1f}")
            with col4:
                spread = year_data["composite_risk"].max() - year_data["composite_risk"].min()
                st.metric("Spread", f"{spread:.1f} pts")

        st.markdown("")

        # Risk ranking bar chart
        sorted_data = year_data.sort_values("composite_risk", ascending=True)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=sorted_data["composite_risk"],
            y=sorted_data["country"],
            orientation="h",
            marker=dict(
                color=sorted_data["composite_risk"],
                colorscale=RISK_SCALE,
                cmin=0, cmax=100,
                colorbar=dict(
                    title=dict(text="Risk", font=dict(color="#6b7280")),
                    tickfont=dict(color="#6b7280"),
                    thickness=12,
                    len=0.6,
                ),
                line=dict(width=0),
            ),
            text=sorted_data["composite_risk"].round(1),
            textposition="outside",
            textfont=dict(color="#8892a0", size=11),
            hovertemplate="%{y}: %{x:.1f}<extra></extra>",
        ))
        apply_dark_layout(fig_bar, height=480)
        fig_bar.update_layout(
            xaxis=dict(range=[0, 105], title="", showgrid=False),
            yaxis=dict(title="", categoryorder="total ascending"),
            bargap=0.25,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Region + distribution side by side
        col_left, col_right = st.columns(2)

        with col_left:
            if "region" in year_data.columns:
                fig_box = px.strip(
                    year_data, x="region", y="composite_risk",
                    color="region",
                    hover_data=["country"],
                    color_discrete_sequence=COUNTRY_COLORS,
                )
                fig_box.update_traces(marker=dict(size=10, opacity=0.8))
                apply_dark_layout(fig_box, height=380)
                fig_box.update_layout(
                    showlegend=False,
                    xaxis_title="", yaxis_title="Risk Score",
                    yaxis=dict(range=[0, 100]),
                )
                st.plotly_chart(fig_box, use_container_width=True)

        with col_right:
            cat_order = ["Very Low", "Low", "Moderate", "High", "Very High"]
            cat_counts = year_data["risk_category"].value_counts().reindex(cat_order, fill_value=0).reset_index()
            cat_counts.columns = ["category", "count"]
            cat_colors = ["#1b5e20", "#388e3c", "#f9a825", "#e65100", "#b71c1c"]
            fig_cat = go.Figure()
            fig_cat.add_trace(go.Bar(
                x=cat_counts["category"],
                y=cat_counts["count"],
                marker=dict(color=cat_colors, line=dict(width=0)),
                text=cat_counts["count"],
                textposition="outside",
                textfont=dict(color="#8892a0"),
            ))
            apply_dark_layout(fig_cat, height=380)
            fig_cat.update_layout(
                xaxis_title="", yaxis_title="Countries",
                yaxis=dict(dtick=1),
                bargap=0.3,
            )
            st.plotly_chart(fig_cat, use_container_width=True)

    # =============== TAB 2: COMPARISON ===============
    with tab2:
        compare_countries = st.multiselect(
            "Select countries to compare",
            options=sorted(year_data["country"].unique()),
            default=sorted(year_data["country"].unique())[:4],
            max_selections=5,
            key="radar_countries",
        )

        if compare_countries and selected_indicators:
            fig_radar = go.Figure()
            for i, country in enumerate(compare_countries):
                row = year_data[year_data["country"] == country]
                if row.empty:
                    continue
                row = row.iloc[0]
                values = [row.get(ind, 0) for ind in selected_indicators]
                values.append(values[0])
                labels = [i.replace("_", " ").title() for i in selected_indicators]
                labels.append(labels[0])

                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=labels,
                    name=country,
                    fill="toself",
                    fillcolor=f"rgba({int(COUNTRY_COLORS[i % len(COUNTRY_COLORS)][1:3], 16)},{int(COUNTRY_COLORS[i % len(COUNTRY_COLORS)][3:5], 16)},{int(COUNTRY_COLORS[i % len(COUNTRY_COLORS)][5:7], 16)},0.1)",
                    line=dict(color=COUNTRY_COLORS[i % len(COUNTRY_COLORS)], width=2),
                ))

            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#0f1520",
                    radialaxis=dict(
                        visible=True, range=[0, 100],
                        gridcolor="#1a2234", tickfont=dict(color="#6b7280"),
                    ),
                    angularaxis=dict(
                        gridcolor="#1a2234",
                        tickfont=dict(color="#8892a0", size=11),
                    ),
                ),
                paper_bgcolor="#0a0e17",
                font=dict(family="Inter, sans-serif", color="#8892a0"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a0")),
                height=520,
                margin=dict(l=60, r=60, t=40, b=40),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Heatmap
        if selected_indicators:
            heatmap_data = year_data.set_index("country")[selected_indicators]
            heatmap_data = heatmap_data.sort_values(
                selected_indicators[0], ascending=False,
            )

            fig_heat = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=[i.replace("_", " ").title() for i in selected_indicators],
                y=heatmap_data.index.tolist(),
                colorscale=RISK_SCALE,
                zmin=0, zmax=100,
                colorbar=dict(
                    title=dict(text="Risk", font=dict(color="#6b7280")),
                    tickfont=dict(color="#6b7280"),
                    thickness=12,
                ),
                hovertemplate="%{y}<br>%{x}: %{z:.1f}<extra></extra>",
                text=np.round(heatmap_data.values, 1),
                texttemplate="%{text}",
                textfont=dict(color="#c8cdd5", size=11),
            ))
            apply_dark_layout(fig_heat, height=480)
            fig_heat.update_layout(
                yaxis=dict(autorange="reversed"),
                xaxis=dict(side="top"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # =============== TAB 3: TIME SERIES ===============
    with tab3:
        ts_metric = st.selectbox(
            "Metric",
            options=["composite_risk"] + selected_indicators,
            format_func=lambda x: x.replace("_", " ").title(),
        )

        ts_countries = st.multiselect(
            "Countries",
            options=sorted(ts_data["country"].unique()),
            default=sorted(ts_data["country"].unique())[:6],
            key="ts_countries",
        )

        if ts_countries and ts_metric:
            plot_data = ts_data[ts_data["country"].isin(ts_countries)].copy()
            plot_data = plot_data.dropna(subset=[ts_metric])

            fig_ts = go.Figure()
            for i, country in enumerate(sorted(ts_countries)):
                cdata = plot_data[plot_data["country"] == country]
                fig_ts.add_trace(go.Scatter(
                    x=cdata["year"], y=cdata[ts_metric],
                    mode="lines+markers",
                    name=country,
                    line=dict(color=COUNTRY_COLORS[i % len(COUNTRY_COLORS)], width=2),
                    marker=dict(size=5),
                    hovertemplate=f"{country}<br>%{{x}}: %{{y:.1f}}<extra></extra>",
                ))

            apply_dark_layout(fig_ts, height=480)
            fig_ts.update_layout(
                hovermode="x unified",
                xaxis_title="",
                yaxis_title=ts_metric.replace("_", " ").title(),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        # Year-over-year change
        st.markdown("")
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

                fig_yoy = go.Figure()
                colors = ["#e15759" if v > 0 else "#59a14f" for v in yoy["change"]]
                fig_yoy.add_trace(go.Bar(
                    x=yoy["change"],
                    y=yoy["country"],
                    orientation="h",
                    marker=dict(color=colors, line=dict(width=0)),
                    text=yoy["change"].round(1),
                    textposition="outside",
                    textfont=dict(color="#8892a0", size=11),
                    hovertemplate="%{y}: %{x:+.1f}<extra></extra>",
                ))
                apply_dark_layout(fig_yoy, height=420)
                fig_yoy.update_layout(
                    xaxis_title=f"Change {prev_year} to {selected_year}",
                    yaxis_title="",
                    xaxis=dict(zeroline=True, zerolinecolor="#2a3a52"),
                )
                st.plotly_chart(fig_yoy, use_container_width=True)

    # =============== TAB 4: DATA ===============
    with tab4:
        display_cols = ["country_code", "country", "year", "composite_risk", "risk_category"] + \
                       [c for c in selected_indicators if c in year_data.columns]

        year_data_display = year_data.copy()
        if "risk_category" not in year_data_display.columns:
            year_data_display["risk_category"] = year_data_display["composite_risk"].apply(get_risk_category)

        available_display_cols = [c for c in display_cols if c in year_data_display.columns]
        display_df = (
            year_data_display[available_display_cols]
            .sort_values("composite_risk", ascending=False)
            .reset_index(drop=True)
        )

        # Format numeric columns for display
        format_cols = [c for c in available_display_cols
                       if c not in ["country_code", "country", "year", "risk_category"]]
        styled = display_df.style.format({c: "{:.1f}" for c in format_cols})
        st.dataframe(styled, use_container_width=True, height=500)

        # Scatter
        if len(selected_indicators) >= 2:
            st.markdown("")
            col_x, col_y = st.columns(2)
            with col_x:
                x_var = st.selectbox("X-axis", selected_indicators, index=0)
            with col_y:
                y_var = st.selectbox("Y-axis", selected_indicators,
                                     index=min(1, len(selected_indicators) - 1))

            scatter_data = year_data.dropna(subset=[x_var, y_var])
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=scatter_data[x_var],
                y=scatter_data[y_var],
                mode="markers+text",
                text=scatter_data["country_code"],
                textposition="top center",
                textfont=dict(color="#8892a0", size=10),
                marker=dict(
                    size=scatter_data["composite_risk"] / 5 + 6,
                    color=scatter_data["composite_risk"],
                    colorscale=RISK_SCALE,
                    cmin=0, cmax=100,
                    colorbar=dict(
                        title=dict(text="Risk", font=dict(color="#6b7280")),
                        tickfont=dict(color="#6b7280"),
                        thickness=12,
                    ),
                    line=dict(width=1, color="#1a2234"),
                ),
                hovertemplate="%{text}<br>" + x_var.replace("_", " ").title() +
                              ": %{x:.1f}<br>" + y_var.replace("_", " ").title() +
                              ": %{y:.1f}<extra></extra>",
            ))
            apply_dark_layout(fig_scatter, height=480)
            fig_scatter.update_layout(
                xaxis_title=x_var.replace("_", " ").title(),
                yaxis_title=y_var.replace("_", " ").title(),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("")
        csv = risk_df[risk_df["country"].isin(selected_countries)].to_csv(index=False)
        st.download_button(
            label="EXPORT CSV",
            data=csv,
            file_name="sovereign_risk_data.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
