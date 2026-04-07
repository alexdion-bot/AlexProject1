"""
Sovereign Risk Index computation module.

Normalizes indicators into 0-100 scores and computes a composite
Sovereign Risk Index for each country/year.

Methodology:
- Each indicator is min-max normalized across all countries for a given year
- Direction is adjusted so that higher score = higher risk
- Composite index is a weighted average of all available sub-indices
"""

import pandas as pd
import numpy as np


# Weights for composite index (must sum to 1.0)
WEIGHTS = {
    "cds_spreads": 0.25,       # Market perception of default risk
    "debt_to_gdp": 0.20,       # Fiscal sustainability
    "fx_reserves_gdp": 0.10,   # External buffer capacity
    "political_risk": 0.15,    # Governance & political stability
    "inflation": 0.15,         # Monetary stability
    "current_account": 0.15,   # External balance
}

# Whether higher raw values mean higher risk
HIGHER_IS_RISKIER = {
    "cds_spreads": True,
    "debt_to_gdp": True,
    "fx_reserves_gdp": False,   # More reserves = less risk
    "political_risk": True,
    "inflation": True,
    "current_account": False,   # Surplus = less risk
}


def normalize_indicator(
    df: pd.DataFrame,
    higher_is_riskier: bool = True,
    group_by_year: bool = True,
) -> pd.DataFrame:
    """
    Min-max normalize an indicator to 0-100 scale.
    If higher_is_riskier=True, higher raw values map to higher scores.
    If False, the scale is inverted.
    """
    df = df.copy()

    if group_by_year and "year" in df.columns:
        scores = []
        for year, group in df.groupby("year"):
            vmin = group["value"].min()
            vmax = group["value"].max()
            if vmax == vmin:
                group = group.copy()
                group["score"] = 50.0
            else:
                group = group.copy()
                group["score"] = (group["value"] - vmin) / (vmax - vmin) * 100
            if not higher_is_riskier:
                group["score"] = 100 - group["score"]
            scores.append(group)
        df = pd.concat(scores, ignore_index=True)
    else:
        vmin = df["value"].min()
        vmax = df["value"].max()
        if vmax == vmin:
            df["score"] = 50.0
        else:
            df["score"] = (df["value"] - vmin) / (vmax - vmin) * 100
        if not higher_is_riskier:
            df["score"] = 100 - df["score"]

    return df


def compute_political_risk(wb_data: dict) -> pd.DataFrame:
    """
    Compute political risk score from World Bank Governance Indicators.
    Combines: Government Effectiveness, Political Stability,
    Rule of Law, Control of Corruption.

    WGI scores range from -2.5 (worst) to 2.5 (best).
    We invert so higher = riskier.
    """
    governance_indicators = [
        "gov_effectiveness", "political_stability",
        "rule_of_law", "control_corruption",
    ]

    frames = []
    for ind in governance_indicators:
        if ind in wb_data:
            df = wb_data[ind].copy()
            df = df.rename(columns={"value": ind})
            frames.append(df[["country_code", "country", "year", ind]])

    if not frames:
        return pd.DataFrame()

    # Merge all governance indicators
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["country_code", "country", "year"], how="outer")

    # Average the governance scores (higher WGI = better governance)
    gov_cols = [c for c in merged.columns if c in governance_indicators]
    merged["value"] = merged[gov_cols].mean(axis=1)

    # Convert: WGI ranges roughly -2.5 to 2.5
    # Transform to 0-100 risk score (invert: lower governance = higher risk)
    merged["value"] = ((2.5 - merged["value"]) / 5.0) * 100
    merged["value"] = merged["value"].clip(0, 100)

    return merged[["country_code", "country", "year", "value"]]


def compute_fx_reserves_to_gdp(wb_data: dict) -> pd.DataFrame:
    """Compute Foreign Exchange Reserves as % of GDP."""
    if "fx_reserves" not in wb_data or "gdp_current" not in wb_data:
        return pd.DataFrame()

    reserves = wb_data["fx_reserves"][["country_code", "country", "year", "value"]].copy()
    reserves = reserves.rename(columns={"value": "reserves"})

    gdp = wb_data["gdp_current"][["country_code", "year", "value"]].copy()
    gdp = gdp.rename(columns={"value": "gdp"})

    merged = reserves.merge(gdp, on=["country_code", "year"], how="inner")
    merged["value"] = (merged["reserves"] / merged["gdp"]) * 100
    merged = merged[merged["gdp"] > 0]

    return merged[["country_code", "country", "year", "value"]]


def build_risk_index(raw_data: dict) -> pd.DataFrame:
    """
    Build the composite Sovereign Risk Index from raw data.

    Returns a DataFrame with columns:
    country_code, country, year, and one column per sub-index + composite.
    """
    # Prepare each sub-index
    sub_indices = {}

    # 1. CDS Spreads
    if "cds_spreads" in raw_data:
        sub_indices["cds_spreads"] = normalize_indicator(
            raw_data["cds_spreads"], higher_is_riskier=True
        )

    # 2. Debt-to-GDP
    if "debt_to_gdp" in raw_data:
        sub_indices["debt_to_gdp"] = normalize_indicator(
            raw_data["debt_to_gdp"], higher_is_riskier=True
        )

    # 3. FX Reserves / GDP
    fx_gdp = compute_fx_reserves_to_gdp(raw_data)
    if not fx_gdp.empty:
        sub_indices["fx_reserves_gdp"] = normalize_indicator(
            fx_gdp, higher_is_riskier=False
        )

    # 4. Political Risk (from governance indicators)
    pol_risk = compute_political_risk(raw_data)
    if not pol_risk.empty:
        # Already on 0-100 scale, higher = riskier
        pol_risk["score"] = pol_risk["value"]
        sub_indices["political_risk"] = pol_risk

    # 5. Inflation
    if "inflation" in raw_data:
        sub_indices["inflation"] = normalize_indicator(
            raw_data["inflation"], higher_is_riskier=True
        )

    # 6. Current Account Balance
    if "current_account" in raw_data:
        sub_indices["current_account"] = normalize_indicator(
            raw_data["current_account"], higher_is_riskier=False
        )

    if not sub_indices:
        return pd.DataFrame()

    # Merge all sub-indices into one DataFrame
    all_scores = None
    for name, df in sub_indices.items():
        score_df = df[["country_code", "country", "year", "score"]].copy()
        score_df = score_df.rename(columns={"score": name})

        if all_scores is None:
            all_scores = score_df
        else:
            all_scores = all_scores.merge(
                score_df, on=["country_code", "country", "year"], how="outer"
            )

    # Compute weighted composite score
    score_cols = [c for c in WEIGHTS if c in all_scores.columns]
    available_weights = {k: WEIGHTS[k] for k in score_cols}
    total_weight = sum(available_weights.values())
    # Renormalize weights to sum to 1
    norm_weights = {k: v / total_weight for k, v in available_weights.items()}

    def weighted_avg(row):
        total = 0
        w_sum = 0
        for col, w in norm_weights.items():
            if pd.notna(row.get(col)):
                total += row[col] * w
                w_sum += w
        return total / w_sum if w_sum > 0 else np.nan

    all_scores["composite_risk"] = all_scores.apply(weighted_avg, axis=1)
    all_scores = all_scores.sort_values(["year", "composite_risk"], ascending=[True, False])

    return all_scores


def get_risk_category(score: float) -> str:
    """Map composite score to risk category."""
    if score >= 75:
        return "Very High"
    elif score >= 55:
        return "High"
    elif score >= 35:
        return "Moderate"
    elif score >= 15:
        return "Low"
    else:
        return "Very Low"
