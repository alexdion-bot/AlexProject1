"""
Data collection module for the Sovereign Risk Dashboard.

Fetches economic indicators from public APIs:
- World Bank API: Debt-to-GDP ratio, Foreign Exchange Reserves
- Simulated CDS spreads (no free real-time CDS API exists)
- Political Risk scores derived from World Bank governance indicators
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional

# 15 countries spanning developed, emerging, and frontier markets
COUNTRIES = {
    "US": "United States",
    "DE": "Germany",
    "JP": "Japan",
    "GB": "United Kingdom",
    "FR": "France",
    "BR": "Brazil",
    "MX": "Mexico",
    "ZA": "South Africa",
    "TR": "Turkey",
    "AR": "Argentina",
    "IN": "India",
    "CN": "China",
    "RU": "Russia",
    "NG": "Nigeria",
    "EG": "Egypt",
}

# World Bank indicator codes
WB_INDICATORS = {
    "debt_to_gdp": "GC.DOD.TOTL.GD.ZS",         # Central government debt, total (% of GDP)
    "fx_reserves": "FI.RES.TOTL.CD",               # Total reserves (includes gold, current US$)
    "gdp_current": "NY.GDP.MKTP.CD",               # GDP (current US$)
    "gov_effectiveness": "GE.EST",                  # Government Effectiveness (WGI)
    "political_stability": "PV.EST",                # Political Stability (WGI)
    "rule_of_law": "RL.EST",                        # Rule of Law (WGI)
    "control_corruption": "CC.EST",                 # Control of Corruption (WGI)
    "inflation": "FP.CPI.TOTL.ZG",                 # Inflation, consumer prices (annual %)
    "current_account": "BN.CAB.XOKA.GD.ZS",        # Current account balance (% of GDP)
}


def fetch_world_bank_data(
    indicator: str,
    countries: Optional[list] = None,
    start_year: int = 2010,
    end_year: int = 2024,
) -> pd.DataFrame:
    """Fetch data from the World Bank API for given indicator and countries."""
    if countries is None:
        countries = list(COUNTRIES.keys())

    country_str = ";".join(countries)
    url = (
        f"https://api.worldbank.org/v2/country/{country_str}/indicator/{indicator}"
        f"?date={start_year}:{end_year}&format=json&per_page=5000"
    )

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if len(data) < 2 or data[1] is None:
            return pd.DataFrame()

        records = []
        for entry in data[1]:
            if entry["value"] is not None:
                records.append({
                    "country_code": entry["countryiso3166alpha2"] if "countryiso3166alpha2" in entry else entry["country"]["id"],
                    "country": entry["country"]["value"],
                    "year": int(entry["date"]),
                    "value": float(entry["value"]),
                })

        return pd.DataFrame(records)

    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"Warning: Failed to fetch {indicator}: {e}")
        return pd.DataFrame()


def fetch_all_wb_indicators(start_year: int = 2010, end_year: int = 2024) -> dict:
    """Fetch all World Bank indicators and return as dict of DataFrames."""
    results = {}
    for name, code in WB_INDICATORS.items():
        print(f"  Fetching {name} ({code})...")
        df = fetch_world_bank_data(code, start_year=start_year, end_year=end_year)
        if not df.empty:
            results[name] = df
    return results


def generate_synthetic_cds_spreads(
    countries: Optional[dict] = None,
    start_year: int = 2010,
    end_year: int = 2024,
) -> pd.DataFrame:
    """
    Generate realistic synthetic CDS spreads.

    Real sovereign CDS data requires Bloomberg/Refinitiv subscriptions.
    We generate plausible spreads based on country risk profiles,
    with realistic trends and volatility.
    """
    if countries is None:
        countries = COUNTRIES

    # Base CDS spread levels (bps) reflecting typical sovereign risk
    base_spreads = {
        "US": 15, "DE": 12, "JP": 25, "GB": 20, "FR": 25,
        "BR": 180, "MX": 120, "ZA": 200, "TR": 350, "AR": 800,
        "IN": 100, "CN": 60, "RU": 250, "NG": 400, "EG": 450,
    }

    np.random.seed(42)
    records = []

    for code, name in countries.items():
        base = base_spreads.get(code, 200)
        for year in range(start_year, end_year + 1):
            # Add trend and cyclical components
            trend = (year - 2015) * (base * 0.02)  # slight drift
            cycle = base * 0.15 * np.sin((year - 2010) * np.pi / 4)
            noise = np.random.normal(0, base * 0.1)

            # Crisis bumps for specific countries/years
            crisis = 0
            if code == "AR" and year in (2018, 2019, 2020):
                crisis = base * 0.5
            if code == "TR" and year in (2018, 2021):
                crisis = base * 0.3
            if code == "RU" and year >= 2022:
                crisis = base * 1.2
            if year == 2020:  # COVID shock
                crisis += base * 0.2

            spread = max(5, base + trend + cycle + noise + crisis)
            records.append({
                "country_code": code,
                "country": name,
                "year": year,
                "value": round(spread, 1),
            })

    return pd.DataFrame(records)


def build_complete_dataset(use_cache: bool = True) -> dict:
    """
    Build the complete dataset, trying live APIs first,
    falling back to cached sample data.
    """
    print("Fetching data from World Bank API...")
    wb_data = fetch_all_wb_indicators()

    print("Generating synthetic CDS spreads...")
    cds_data = generate_synthetic_cds_spreads()

    dataset = {**wb_data, "cds_spreads": cds_data}

    # Report what we got
    for name, df in dataset.items():
        if not df.empty:
            print(f"  ✓ {name}: {len(df)} records, {df['country_code'].nunique()} countries")
        else:
            print(f"  ✗ {name}: no data")

    return dataset
