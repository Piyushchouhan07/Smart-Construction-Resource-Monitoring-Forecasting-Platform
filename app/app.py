"""
Smart Construction AI Dashboard
BIM (Revit) + IoT Simulation + Machine Learning, wired end-to-end.

Run with:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `src` importable whether Streamlit is launched from the repo root
# (streamlit run app/app.py) or from inside app/.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.iot_simulation import load_revit_data, simulate_iot_data
from src.train_model import FEATURES, train_and_compare, pick_best_model, forecast_next_days

st.set_page_config(page_title="Smart Construction AI", layout="wide")
st.title("🏗️ Smart Construction AI Dashboard")
st.markdown("BIM (Revit) + IoT Simulation + Machine Learning")

# -----------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")
project_days = st.sidebar.slider("Simulation Days per Zone", 30, 200, 150)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)

st.subheader("📂 Upload Revit Quantity-Takeoff CSV")
st.caption(
    "Expected columns: `Material: Volume` (e.g. `100.54 m³`) and, optionally, "
    "`Comments` (used as the zone name)."
)
file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    # Revit exports use latin1 encoding because of the 'm³' symbol.
    df_revit = pd.read_csv(file, encoding="latin1")
    st.write("**Preview of uploaded Revit data:**")
    st.dataframe(df_revit.head())

    required_cols = {"Material: Volume"}
    missing = required_cols - set(df_revit.columns)
    if missing:
        st.error(f"Missing required column(s): {', '.join(missing)}. Please check your export.")
        st.stop()

    # -----------------------------------------------------------------
    # 1. SIMULATE IoT DATA
    # -----------------------------------------------------------------
    with st.spinner("Simulating site conditions..."):
        iot_df = simulate_iot_data(df_revit, project_days=project_days, seed=int(seed))

    if iot_df.empty:
        st.error(
            "No usable rows found — every row was missing a parsable 'Material: Volume' value."
        )
        st.stop()

    st.subheader("📡 Simulated IoT Data")
    st.dataframe(iot_df.head(20))
    st.line_chart(iot_df.pivot_table(index="Date", columns="Zone", values="Daily_Usage_m3", aggfunc="sum"))

    # -----------------------------------------------------------------
    # 2. TRAIN & COMPARE MODELS
    # -----------------------------------------------------------------
    st.subheader("🤖 Model Comparison")
    with st.spinner("Training RandomForest, XGBoost, and LightGBM..."):
        trained_models, results = train_and_compare(iot_df)

    result_df = pd.DataFrame({
        "Model": list(results.keys()),
        "R2 Score": [m["r2"] for m in results.values()],
        "MAE": [m["mae"] for m in results.values()],
    })
    st.dataframe(result_df)
    st.bar_chart(result_df.set_index("Model")["R2 Score"])

    best_name, best_model = pick_best_model(trained_models, results)
    st.success(f"🏆 Best Model: {best_name}")

    # -----------------------------------------------------------------
    # 3. KPI CARDS
    # -----------------------------------------------------------------
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Daily Usage", f"{iot_df['Daily_Usage_m3'].mean():.2f} m³")
    col2.metric("Max Daily Usage", f"{iot_df['Daily_Usage_m3'].max():.2f} m³")
    col3.metric("Total CO₂ Emitted", f"{iot_df['CO2_Emission_kg'].sum():,.0f} kg")
    col4.metric("Days at Shortage Risk", int(iot_df["Shortage_Risk"].sum()))

    # -----------------------------------------------------------------
    # 4. WHAT-IF PREDICTION
    # -----------------------------------------------------------------
    st.subheader("🔮 Predict Usage for a Custom Scenario")
    zones = sorted(iot_df["Zone"].unique().tolist())

    c1, c2, c3 = st.columns(3)
    zone = c1.selectbox("Zone", zones)
    labor = c1.slider("Labor Count", 5, 30, 15)
    rain = c2.selectbox("Weather", ["Sun", "Rain"])
    truck_min = c2.slider("Truck Travel Time (min)", 5, 120, 30)
    quality = c3.selectbox("Concrete Quality", ["GOOD", "WARNING", "REJECTED"])
    scenario_date = c3.date_input("Date", value=pd.Timestamp(iot_df["Date"].max()) + pd.Timedelta(days=1))

    scenario = pd.DataFrame([{
        "Date": pd.to_datetime(scenario_date),
        "Zone": zone,
        "Labor_Count": labor,
        "Weather_Rain": 1 if rain == "Rain" else 0,
        "Truck_Travel_Min": truck_min,
        "Concrete_Quality": quality,
    }])
    scenario["day"] = scenario["Date"].dt.day
    scenario["month"] = scenario["Date"].dt.month
    scenario["weekday"] = scenario["Date"].dt.weekday

    pred = best_model.predict(scenario[FEATURES])[0]
    st.metric("Predicted Usage", f"{pred:.2f} m³")

    # -----------------------------------------------------------------
    # 5. 7-DAY FORECAST
    # -----------------------------------------------------------------
    st.subheader("📅 7-Day Forecast")
    forecast = forecast_next_days(best_model, iot_df, zones=zones, days=7, seed=int(seed))
    forecast_totals = forecast.groupby("Date")["Predicted_Daily_Usage_m3"].sum()
    st.line_chart(forecast_totals)
    st.dataframe(forecast[["Date", "Zone", "Predicted_Daily_Usage_m3"]])
    st.metric("Total Concrete Needed (Next 7 Days)", f"{forecast['Predicted_Daily_Usage_m3'].sum():.2f} m³")

else:
    st.info("👆 Upload a Revit quantity-takeoff CSV to start.")

# -----------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------
st.markdown("---")
st.markdown("🚀 Developed by Piyush & Team | Smart Construction AI")
