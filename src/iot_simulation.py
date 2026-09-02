"""
IoT data simulation.

Turns a Revit quantity-takeoff export into a synthetic day-by-day
construction-site dataset (labor, weather, truck logistics, concrete
quality, usage, CO2, and shortage risk) that the ML pipeline trains on.

This is the same logic that lived in notebooks/01_IOT_Simulation.ipynb,
pulled out into a function so both the notebook and the Streamlit app
(app/app.py) can call the exact same code instead of drifting apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTPUT_COLUMNS = [
    "Date", "Zone", "Material", "Labor_Count", "Weather_Rain",
    "Truck_Travel_Min", "Concrete_Quality", "Daily_Usage_m3",
    "Cumulative_Usage_m3", "CO2_Emission_kg", "Shortage_Risk",
]


def load_revit_data(path: str) -> pd.DataFrame:
    """Read a Revit quantity-takeoff CSV.

    Revit exports volume units with the 'm³' character, which is not
    valid ASCII/UTF-8 in every export encoding, so we read as latin1.
    """
    return pd.read_csv(path, encoding="latin1")


def simulate_iot_data(
    df_revit: pd.DataFrame,
    project_days: int = 150,
    start_date: str = "2025-11-01",
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate a day-by-day IoT dataset from Revit quantity-takeoff rows.

    Parameters
    ----------
    df_revit : DataFrame with at least 'Material: Volume' and, optionally,
        'Comments' (used as the zone label).
    project_days : number of days to simulate per Revit row/zone.
    start_date : first simulated calendar date.
    seed : optional RNG seed for reproducible simulations.

    Returns
    -------
    DataFrame with columns OUTPUT_COLUMNS.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=project_days, freq="D")
    rows = []

    for _, row in df_revit.iterrows():
        try:
            zone = str(row.get("Comments", "Unknown Zone"))
            vol_str = str(row["Material: Volume"])
            total_volume = float(
                vol_str.replace("m\u00b3", "").replace("m3", "").strip()
            )
        except (KeyError, ValueError):
            continue  # skip rows we can't parse a volume out of

        avg_daily_usage = total_volume / project_days
        cumulative_usage = 0.0

        for current_date in dates:
            # 1. Labor & weather
            labor = int(rng.integers(5, 30))
            weather = int(rng.choice([0, 1], p=[0.85, 0.15]))  # 0=Sun, 1=Rain

            # 2. Truck speed depends on weather
            speed = int(rng.integers(10, 20)) if weather == 1 else int(rng.integers(30, 45))
            travel_time_min = (15.0 / speed) * 60

            # 3. Concrete quality depends on travel/setting time
            if travel_time_min > 90:
                quality = "REJECTED"
            elif travel_time_min > 60:
                quality = "WARNING"
            else:
                quality = "GOOD"

            # 4. Usage: rain halts pours, labor boosts productivity
            if weather == 1:
                daily_used = 0.0
            else:
                productivity = 1 + (labor / 50)
                daily_used = avg_daily_usage * productivity

            # 5. Sustainability: CO2 per m3 poured
            co2 = daily_used * 288

            # 6. Shortage risk once 95% of the material budget is used
            cumulative_usage += daily_used
            risk = 1 if cumulative_usage > (total_volume * 0.95) else 0

            rows.append([
                current_date, zone, "Concrete", labor, weather,
                int(travel_time_min), quality, round(daily_used, 2),
                round(cumulative_usage, 2), round(co2, 2), risk,
            ])

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
