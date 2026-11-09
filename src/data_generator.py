"""
OptiChain AI - Global Supply Chain Logistics & Disruption Benchmark Generator.
Generates realistic multi-modal freight shipment datasets with physical transit dynamics,
port dwell congestion, geopolitical/weather shocks, and customs clearance holds.
"""

import random
from typing import List, Dict, Any
import numpy as np
import pandas as pd


ORIGIN_PORTS = [
    "Shanghai (CNSHA)", "Rotterdam (NLRTM)", "Los Angeles (USLAX)",
    "Singapore (SGSIN)", "Hamburg (DEHAM)", "Busan (KRPUS)",
    "Dubai (AEDXB)", "Santos (BRSSZ)"
]

DESTINATION_PORTS = [
    "Long Beach (USLGB)", "Antwerp (BEANR)", "Tokyo (JPTYO)",
    "New York (USNYC)", "Felixstowe (GBFXT)", "Jebel Ali (AEJEA)"
]

TRANSPORT_MODES = [
    ("Ocean Freight", 18.0, 6.0, 0.65),     # mode, base_days, std_days, mode_share
    ("Air Cargo", 3.0, 1.2, 0.15),
    ("Rail Intermodal", 9.0, 2.5, 0.10),
    ("Cross-Border Trucking", 5.0, 1.8, 0.10)
]


def generate_supply_chain_dataset(n_samples: int = 3500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a high-fidelity synthetic global logistics dataset with physically consistent
    correlations between carrier reliability, weather shocks, port congestion, and SLA delays.
    """
    np.random.seed(random_state)
    random.seed(random_state)

    records: List[Dict[str, Any]] = []

    modes, base_transits, std_transits, probs = zip(*TRANSPORT_MODES)

    for i in range(n_samples):
        shipment_id = f"OPT-SHP-{i+1:05d}"
        origin = random.choice(ORIGIN_PORTS)
        dest = random.choice([d for d in DESTINATION_PORTS if d[:3] != origin[:3]])

        mode_idx = np.random.choice(len(modes), p=probs)
        mode = modes[mode_idx]
        nominal_days = max(1.5, np.random.normal(base_transits[mode_idx], std_transits[mode_idx]))

        # Operational metrics
        carrier_rating = round(float(np.random.uniform(2.2, 5.0)), 2)
        supplier_reliability = round(float(np.random.beta(5, 1.5)), 3) # skewed towards reliable
        port_congestion = round(float(np.random.exponential(scale=2.2) + 1.0), 2)
        port_congestion = min(10.0, port_congestion)

        weather_risk = round(float(np.random.beta(1.5, 4.0)), 3) # occasional severe storms
        customs_exam = 1 if np.random.rand() < 0.18 else 0
        customs_dwell_hrs = round(float(np.random.exponential(scale=18.0) if customs_exam else np.random.exponential(scale=3.0)), 1)
        customs_dwell_hrs = min(144.0, customs_dwell_hrs)

        cargo_val = round(float(np.random.lognormal(mean=11.2, sigma=0.85)), 2)
        is_perishable = 1 if np.random.rand() < 0.14 else 0
        container_teu = int(np.random.choice([1, 2, 4, 8, 12, 20], p=[0.45, 0.25, 0.15, 0.08, 0.05, 0.02]))
        bunker_fuel_index = round(float(np.random.normal(118.0, 12.0)), 1)

        # Multi-factorial delay calculation (Physics & Operations Research model)
        congestion_delay = max(0.0, (port_congestion - 3.5) * 0.75) if port_congestion > 3.5 else 0.0
        weather_delay = max(0.0, (weather_risk - 0.35) * 8.5) if weather_risk > 0.35 else 0.0
        carrier_friction = max(0.0, (4.0 - carrier_rating) * 1.6)
        supplier_delay = max(0.0, (0.75 - supplier_reliability) * 5.0) if supplier_reliability < 0.75 else 0.0
        customs_delay_days = customs_dwell_hrs / 24.0

        random_noise = np.random.normal(0.0, 0.8)
        total_delay = congestion_delay + weather_delay + carrier_friction + supplier_delay + customs_delay_days + random_noise
        total_delay = max(0.0, total_delay)

        actual_days = round(nominal_days + total_delay, 2)
        delay_days = round(max(0.0, actual_days - nominal_days), 2)

        # Binary threshold: delayed if arrival exceeds nominal by > 1.5 days
        is_delayed = 1 if delay_days >= 1.5 else 0

        # Severity classification
        if delay_days < 1.5:
            severity = "ON_TIME"
        elif delay_days < 4.0:
            severity = "MINOR_DELAY"
        elif delay_days < 8.0:
            severity = "MODERATE_DELAY"
        else:
            severity = "CRITICAL_DISRUPTION"

        records.append({
            "shipment_id": shipment_id,
            "origin_port": origin,
            "destination_port": dest,
            "transport_mode": mode,
            "nominal_transit_days": round(nominal_days, 2),
            "carrier_rating": carrier_rating,
            "supplier_reliability": supplier_reliability,
            "port_congestion_index": port_congestion,
            "weather_risk_score": weather_risk,
            "customs_inspection": customs_exam,
            "customs_dwell_hours": customs_dwell_hrs,
            "cargo_value_usd": cargo_val,
            "is_perishable": is_perishable,
            "container_teu": container_teu,
            "bunker_fuel_index": bunker_fuel_index,
            "actual_transit_days": actual_days,
            "delay_days": delay_days,
            "is_delayed": is_delayed,
            "delay_severity": severity
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df_sample = generate_supply_chain_dataset(50)
    print("Dataset generated successfully:")
    print(df_sample[["shipment_id", "transport_mode", "nominal_transit_days", "actual_transit_days", "delay_severity"]].head())
    print("\nClass distribution:")
    print(df_sample["delay_severity"].value_counts())
