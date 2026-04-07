"""
Script to generate sample/cached data for offline demo.
Run once: python generate_sample_data.py
"""

import json
import os
import numpy as np
import pandas as pd

COUNTRIES = {
    "US": "United States", "DE": "Germany", "JP": "Japan",
    "GB": "United Kingdom", "FR": "France", "BR": "Brazil",
    "MX": "Mexico", "ZA": "South Africa", "TR": "Turkey",
    "AR": "Argentina", "IN": "India", "CN": "China",
    "RU": "Russia", "NG": "Nigeria", "EG": "Egypt",
}

np.random.seed(42)
START, END = 2010, 2024


def gen_indicator(base_values, volatility=0.05, trend=0.0):
    records = []
    for code, name in COUNTRIES.items():
        base = base_values.get(code, 50)
        for year in range(START, END + 1):
            t = (year - START) * trend
            noise = np.random.normal(0, abs(base) * volatility)
            val = base + t + noise
            records.append({"country_code": code, "country": name, "year": year, "value": round(val, 2)})
    return records


# Debt-to-GDP (%)
debt_gdp_base = {
    "US": 105, "DE": 65, "JP": 230, "GB": 85, "FR": 98,
    "BR": 75, "MX": 45, "ZA": 55, "TR": 30, "AR": 85,
    "IN": 70, "CN": 55, "RU": 15, "NG": 25, "EG": 90,
}
debt_to_gdp = gen_indicator(debt_gdp_base, volatility=0.03, trend=1.5)

# FX Reserves (USD)
fx_base = {
    "US": 140e9, "DE": 200e9, "JP": 1250e9, "GB": 170e9, "FR": 160e9,
    "BR": 360e9, "MX": 190e9, "ZA": 50e9, "TR": 80e9, "AR": 40e9,
    "IN": 400e9, "CN": 3200e9, "RU": 460e9, "NG": 35e9, "EG": 35e9,
}
fx_reserves = gen_indicator(fx_base, volatility=0.08)

# GDP (USD)
gdp_base = {
    "US": 20e12, "DE": 4e12, "JP": 5e12, "GB": 2.8e12, "FR": 2.7e12,
    "BR": 1.8e12, "MX": 1.3e12, "ZA": 0.4e12, "TR": 0.8e12, "AR": 0.5e12,
    "IN": 2.5e12, "CN": 14e12, "RU": 1.5e12, "NG": 0.45e12, "EG": 0.35e12,
}
gdp_current = gen_indicator(gdp_base, volatility=0.04, trend=0.02e12)

# Governance indicators (WGI scale: -2.5 to +2.5)
gov_base = {
    "US": 1.5, "DE": 1.7, "JP": 1.5, "GB": 1.6, "FR": 1.4,
    "BR": -0.2, "MX": -0.3, "ZA": 0.1, "TR": -0.3, "AR": -0.4,
    "IN": -0.1, "CN": 0.3, "RU": -0.5, "NG": -1.0, "EG": -0.7,
}
gov_effectiveness = gen_indicator(gov_base, volatility=0.05)
political_stability = gen_indicator(
    {k: v - 0.3 for k, v in gov_base.items()}, volatility=0.08
)
rule_of_law = gen_indicator(
    {k: v + 0.1 for k, v in gov_base.items()}, volatility=0.04
)
control_corruption = gen_indicator(
    {k: v - 0.1 for k, v in gov_base.items()}, volatility=0.06
)

# Inflation (%)
inflation_base = {
    "US": 2.0, "DE": 1.5, "JP": 0.5, "GB": 2.0, "FR": 1.5,
    "BR": 6.0, "MX": 4.5, "ZA": 5.0, "TR": 15.0, "AR": 40.0,
    "IN": 5.5, "CN": 2.5, "RU": 6.0, "NG": 12.0, "EG": 13.0,
}
inflation = gen_indicator(inflation_base, volatility=0.15)

# Current account (% of GDP)
ca_base = {
    "US": -2.5, "DE": 7.0, "JP": 3.5, "GB": -3.5, "FR": -0.5,
    "BR": -2.0, "MX": -1.5, "ZA": -3.0, "TR": -5.0, "AR": -2.5,
    "IN": -1.5, "CN": 2.0, "RU": 5.0, "NG": -1.0, "EG": -3.5,
}
current_account = gen_indicator(ca_base, volatility=0.2)

# CDS spreads (bps)
from src.data_collection import generate_synthetic_cds_spreads
cds = generate_synthetic_cds_spreads()

# Assemble and save
data = {
    "debt_to_gdp": debt_to_gdp,
    "fx_reserves": fx_reserves,
    "gdp_current": gdp_current,
    "gov_effectiveness": gov_effectiveness,
    "political_stability": political_stability,
    "rule_of_law": rule_of_law,
    "control_corruption": control_corruption,
    "inflation": inflation,
    "current_account": current_account,
    "cds_spreads": cds.to_dict(orient="records"),
}

os.makedirs("data", exist_ok=True)
with open("data/cached_data.json", "w") as f:
    json.dump(data, f)

print(f"Sample data saved to data/cached_data.json")
for k, v in data.items():
    count = len(v) if isinstance(v, list) else len(v)
    print(f"  {k}: {count} records")
