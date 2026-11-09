"""
OptiChain AI — Executive Supply Chain Control Tower & Inventory Optimization Cockpit.
Built with Streamlit.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.engine import OptiChainEngine
from src.data_generator import ORIGIN_PORTS, DESTINATION_PORTS


st.set_page_config(
    page_title="OptiChain AI — Supply Chain Resilience & Inventory Cockpit",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for industrial executive dashboard
st.markdown("""
<style>
    .metric-container {
        background-color: #0F172A;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
        border-left: 5px solid #0EA5E9;
    }
    .badge-critical {
        background-color: #7F1D1D;
        color: #F87171;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-high {
        background-color: #78350F;
        color: #FBBF24;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-low {
        background-color: #064E3B;
        color: #34D399;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_optichain_engine():
    model_path = os.path.join(BASE_DIR, "artifacts", "optichain_engine.joblib")
    if os.path.exists(model_path):
        return OptiChainEngine.load(model_path)
    return OptiChainEngine()


engine = load_optichain_engine()

# Sidebar
with st.sidebar:
    st.image("https://img.shields.io/badge/OptiChain-Resilience%20v1.0-0288D1?style=for-the-badge&logo=cargo", use_container_width=True)
    st.title("🌐 Global Logistics Hub")
    st.markdown("Autonomous supply chain disruption intelligence & prescriptive inventory engine.")
    st.divider()

    st.subheader("⚙️ Global Control Tower Settings")
    default_sla = st.slider("Target Fulfillment SLA", 80.0, 99.9, 95.0, step=0.5,
                            help="Customer order service level used for safety stock sizing.")
    holding_cost_rate = st.slider("Annual Inventory Holding Cost Rate (%)", 10.0, 40.0, 20.0, step=1.0) / 100.0

    st.divider()
    st.markdown("### 📊 Operational Telemetry")
    st.success("🟢 Disruption Engine: ACTIVE")
    st.info("⏱️ Quantile Model: p10 / p50 / p90 / p99")
    st.caption("Engineered by **Salman Ali** | Junior Agentic AI Engineer")

# Header
st.title("🚢 OptiChain AI — Supply Chain Resilience & Inventory Optimization")
st.markdown(
    "Enterprise decision intelligence for multi-modal logistics: **Freight Delay Classification**, "
    "**Probabilistic Lead-Time Quantile Regression**, **Dynamic Safety Stock Sizing**, and **Geopolitical Shock Stress-Testing**."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚢 Freight Disruption Predictor",
    "📦 Multi-Echelon Inventory Optimizer",
    "🌪️ Supply Chain Shock Simulator",
    "⏱️ Probabilistic Lead Time Forecaster",
    "📈 Certified Model Diagnostics"
])

# ---------------------------------------------------------
# TAB 1: FREIGHT DISRUPTION PREDICTOR
# ---------------------------------------------------------
with tab1:
    st.subheader("🚢 Multi-Modal Freight Disruption & SLA Risk Predictor")
    st.markdown("Assess incoming shipment telemetry for port dwell delays, customs clearance bottlenecks, and carrier SLA violations.")

    col_l, col_r = st.columns(2)
    with col_l:
        origin = st.selectbox("Origin Port:", ORIGIN_PORTS, index=0)
        dest = st.selectbox("Destination Port:", DESTINATION_PORTS, index=0)
        mode = st.selectbox("Transport Mode:", ["Ocean Freight", "Air Cargo", "Rail Intermodal", "Cross-Border Trucking"], index=0)
        nominal_days = st.number_input("Nominal Scheduled Transit (Days):", min_value=1.0, max_value=60.0, value=18.0, step=0.5)
        carrier_rating = st.slider("Carrier On-Time Reliability Rating (1-5):", 1.0, 5.0, 4.1, step=0.1)
        supplier_rel = st.slider("Manufacturing Supplier Reliability (0.0-1.0):", 0.1, 1.0, 0.90, step=0.02)
        cargo_val = st.number_input("Commercial Cargo Invoice Value (USD):", min_value=1000.0, max_value=5000000.0, value=180000.0, step=10000.0)

    with col_r:
        congestion = st.slider("Port Dwell Congestion Index (1-10):", 1.0, 10.0, 3.8, step=0.2)
        weather_risk = st.slider("Route Weather Severity Score (0-1):", 0.0, 1.0, 0.22, step=0.02)
        customs_exam = st.checkbox("Customs Intensive Examination Triggered", value=False)
        customs_dwell = st.slider("Customs Clearance Dwell (Hours):", 0.0, 144.0, 6.0 if not customs_exam else 48.0, step=2.0)
        is_perishable = st.checkbox("Cold-Chain / Temperature Sensitive Cargo", value=False)
        container_teu = st.selectbox("Container Volume (TEU):", [1, 2, 4, 8, 12, 20], index=1)
        bunker_fuel = st.slider("Bunker Fuel Price Index:", 80.0, 200.0, 118.0, step=1.0)

    shipment_payload = {
        "origin_port": origin,
        "destination_port": dest,
        "transport_mode": mode,
        "nominal_transit_days": float(nominal_days),
        "carrier_rating": float(carrier_rating),
        "supplier_reliability": float(supplier_rel),
        "port_congestion_index": float(congestion),
        "weather_risk_score": float(weather_risk),
        "customs_inspection": 1 if customs_exam else 0,
        "customs_dwell_hours": float(customs_dwell),
        "cargo_value_usd": float(cargo_val),
        "is_perishable": 1 if is_perishable else 0,
        "container_teu": int(container_teu),
        "bunker_fuel_index": float(bunker_fuel)
    }

    if st.button("⚡ Predict Shipment Delay & Quantiles", type="primary", use_container_width=True):
        pred = engine.predict_shipment(shipment_payload)

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Disruption Probability", f"{pred['disruption_probability'] * 100:.1f}%")
        m2.metric("Predicted Status", "🚨 DELAY EXPECTED" if pred["is_delayed_predicted"] else "✅ ON TIME")
        m3.metric("Risk Classification", pred["risk_tier"])
        m4.metric("Inference Latency", f"{pred['latency_ms']:.2f} ms")

        st.info(f"💡 **Prescriptive Mitigation Strategy:** `{pred['mitigation_strategy']}`")

        st.subheader("⏱️ Probabilistic Transit Time Quantiles")
        q = pred["lead_time_quantiles_days"]
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("p10 (Optimistic)", f"{q['p10']} days")
        qc2.metric("p50 (Expected Median)", f"{q['p50']} days", delta=f"{pred['expected_delay_days']:.1f}d delay" if pred['expected_delay_days'] > 0 else "On schedule")
        qc3.metric("p90 (Conservative Cutoff)", f"{q['p90']} days")
        qc4.metric("p99 (Extreme Tail Risk)", f"{q['p99']} days")

# ---------------------------------------------------------
# TAB 2: MULTI-ECHELON INVENTORY OPTIMIZER
# ---------------------------------------------------------
with tab2:
    st.subheader("📦 Prescriptive Multi-Echelon Inventory Optimization (MEIO)")
    st.markdown("Calculate dynamic safety stock buffers, reorder points, and optimal order batch sizes factoring in both lead-time and demand variance.")

    inv_c1, inv_c2 = st.columns(2)
    with inv_c1:
        st.markdown("#### ⏱️ Transit Lead Time Distribution")
        lead_mean = st.number_input("Mean Lead Time (Days):", min_value=1.0, max_value=90.0, value=18.5, step=0.5)
        lead_p90 = st.number_input("Conservative Lead Time p90 (Days):", min_value=1.0, max_value=120.0, value=24.0, step=0.5)
        sla_target = st.slider("Target Service Level SLA (%):", 80.0, 99.9, float(default_sla), step=0.5) / 100.0

    with inv_c2:
        st.markdown("#### 📈 Demand & Cost Economics")
        demand_mean = st.number_input("Daily Demand Mean (Units):", min_value=1.0, max_value=5000.0, value=120.0, step=10.0)
        demand_std = st.number_input("Daily Demand Volatility / Std Dev (Units):", min_value=0.5, max_value=1000.0, value=25.0, step=2.0)
        unit_cost = st.number_input("Unit COGS Value (USD):", min_value=1.0, max_value=10000.0, value=65.0, step=5.0)
        order_cost = st.number_input("Order Setup Cost (USD per order):", min_value=10.0, max_value=5000.0, value=350.0, step=25.0)

    if st.button("📊 Calculate Optimal Inventory Policy", type="primary", use_container_width=True):
        policy = engine.optimize_inventory(
            mean_lead_time_days=float(lead_mean),
            lead_time_p90_days=float(lead_p90),
            daily_demand_mean=float(demand_mean),
            daily_demand_std=float(demand_std),
            order_cost_usd=float(order_cost),
            holding_cost_annual_pct=float(holding_cost_rate),
            unit_cost_usd=float(unit_cost),
            service_level=float(sla_target)
        )

        st.divider()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Safety Stock Buffer", f"{policy['safety_stock_units']:,.1f} units")
        k2.metric("Dynamic Reorder Point (ROP)", f"{policy['reorder_point_units']:,.1f} units")
        k3.metric("Economic Order Qty (EOQ)", f"{policy['economic_order_quantity_units']:,.1f} units")
        k4.metric("Orders Per Year", f"{policy['orders_per_year']} cycles")

        st.markdown("#### 💰 Annual Inventory Cost Breakdown:")
        c1, c2, c3 = st.columns(3)
        c1.metric("Annual Holding Cost", f"${policy['annual_holding_cost_usd']:,.2f}")
        c2.metric("Annual Ordering Cost", f"${policy['annual_ordering_cost_usd']:,.2f}")
        c3.metric("Total Supply Chain Cost", f"${policy['total_inventory_cost_usd']:,.2f}")

        st.caption(f"Estimated Lead Time Standard Deviation: `{policy['lead_time_volatility_days']:.2f} days` | Service Level: `{policy['service_level_target_pct']}%`")

# ---------------------------------------------------------
# TAB 3: SHOCK SIMULATOR
# ---------------------------------------------------------
with tab3:
    st.subheader("🌪️ What-If Geopolitical & Climate Supply Chain Shock Simulator")
    st.markdown("Stress-test freight resilience against black-swan macro events before disruptions cascade down your supply tiers.")

    scenario_choice = st.selectbox(
        "Select Black-Swan Scenario:",
        [
            "SUEZ_CANAL_BLOCKAGE",
            "PORT_LABOR_STRIKE",
            "CATEGORY_5_TYPHOON",
            "SUPPLIER_BANKRUPTCY"
        ]
    )

    scenario_descriptions = {
        "SUEZ_CANAL_BLOCKAGE": "Chokepoint maritime blockage forcing Cape of Good Hope circumnavigation (+12 days nominal transit, +4.5 port congestion, +35 fuel index).",
        "PORT_LABOR_STRIKE": "Major stevedore strike across key destination terminal (congestion jumps to 9.8/10, customs dwell escalates to 120 hours).",
        "CATEGORY_5_TYPHOON": "Catastrophic typhoon season hitting transpacific corridor (weather risk jumps to 0.95, carrier reliability degrades by -1.5 rating).",
        "SUPPLIER_BANKRUPTCY": "Primary component tier-1 supplier enters insolvency (supplier reliability plunges to 0.15, severe manufacturing bottlenecks)."
    }

    st.info(f"📌 **Scenario Parameters:** {scenario_descriptions[scenario_choice]}")

    if st.button("🔥 Run Stress Test Simulation", type="primary"):
        sim_res = engine.simulate_shock(shipment_payload, scenario_choice)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Baseline Disruption Risk", f"{sim_res['baseline_probability'] * 100:.1f}%")
        with col2:
            st.metric("Shocked Disruption Risk", f"{sim_res['shocked_probability'] * 100:.1f}%", delta=f"+{sim_res['delta_disruption_probability']*100:.1f}%", delta_color="inverse")
        with col3:
            st.metric("Median Lead Time Delay", f"{sim_res['shocked_median_lead_time']} days", delta=f"+{sim_res['delta_lead_time_days']:.1f} days", delta_color="inverse")

        st.error(f"🚨 **Escalated Risk Classification:** `{sim_res['shocked_risk_tier']}`")
        st.warning(f"🛡️ **Prescriptive Emergency Action:** `{sim_res['shocked_mitigation']}`")

# ---------------------------------------------------------
# TAB 4: QUANTILE LEAD TIME FORECASTER
# ---------------------------------------------------------
with tab4:
    st.subheader("⏱️ Probabilistic Lead Time Quantile Forecasting")
    st.markdown("Why point forecasts fail in supply chain: point estimates hide tail risks. Multi-quantile regression predicts the entire arrival density.")

    img_quant = os.path.join(BASE_DIR, "artifacts", "lead_time_quantiles.png")
    if os.path.exists(img_quant):
        st.image(Image.open(img_quant), caption="Lead Time Quantile Fan & Safety Stock Sizing Curves", use_container_width=True)

    st.markdown("""
    ### 🔬 Theoretical Formulation
    - **Quantile Pinball Loss:** $\mathcal{L}_q(y, \hat{y}) = \max(q(y - \hat{y}), (1 - q)(\hat{y} - y))$
    - Sizing inventory to $p_{90}$ lead times guarantees coverage across $90\%$ of maritime volatility scenarios without over-capitalizing working capital.
    """)

# ---------------------------------------------------------
# TAB 5: CERTIFIED BENCHMARK DIAGNOSTICS
# ---------------------------------------------------------
with tab5:
    st.subheader("📈 Certified Model Diagnostics & Offline Evaluation")
    st.markdown("Offline benchmark metrics evaluated on 700 held-out global freight shipments.")

    metrics_path = os.path.join(BASE_DIR, "artifacts", "benchmark_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            bench = json.load(f)

        c = bench["classification_metrics"]
        r = bench["regression_lead_time_metrics"]
        lat = bench["latency_benchmarks_ms"]

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Classification Accuracy", f"{c['accuracy'] * 100:.2f}%")
        b2.metric("Precision", f"{c['precision'] * 100:.2f}%")
        b3.metric("Recall", f"{c['recall'] * 100:.2f}%")
        b4.metric("ROC-AUC", f"{c['roc_auc']:.4f}")
        b5.metric("Lead Time MAE", f"{r['median_mae_days']} days")

        st.divider()

    img_diag = os.path.join(BASE_DIR, "artifacts", "model_diagnostics.png")
    if os.path.exists(img_diag):
        st.image(Image.open(img_diag), caption="OptiChain Model Performance Diagnostics", use_container_width=True)
