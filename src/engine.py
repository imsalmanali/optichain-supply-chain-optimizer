"""
OptiChain AI - Master Supply Chain Intelligence Engine.
Coordinates feature transformations, disruption classification, lead-time quantile forecasting,
and multi-echelon inventory optimization into an end-to-end operational pipeline.
"""

import time
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

from src.features import LogisticsFeatureEngineer
from src.models import DisruptionClassifier, LeadTimeQuantileForecaster, InventoryOptimizer


class OptiChainEngine:
    """
    Unified enterprise platform for supply chain resilience and multi-echelon inventory optimization.
    """

    def __init__(
        self,
        feature_engineer: Optional[LogisticsFeatureEngineer] = None,
        classifier: Optional[DisruptionClassifier] = None,
        forecaster: Optional[LeadTimeQuantileForecaster] = None,
        optimizer: Optional[InventoryOptimizer] = None
    ):
        self.feature_engineer = feature_engineer or LogisticsFeatureEngineer()
        self.classifier = classifier or DisruptionClassifier()
        self.forecaster = forecaster or LeadTimeQuantileForecaster()
        self.optimizer = optimizer or InventoryOptimizer()

    def predict_shipment(self, shipment_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict delivery delay probability, risk classification, and lead time distribution.
        """
        start = time.perf_counter()
        df_single = pd.DataFrame([shipment_dict])

        # Feature transformation
        X_trans = self.feature_engineer.transform(df_single)

        # 1. Disruption Classification
        proba = float(self.classifier.predict_proba(X_trans)[0][1])
        is_delayed = bool(proba >= 0.50)
        risk_tier, mitigation_strategy = self.classifier.assess_risk(proba)

        # 2. Lead Time Quantiles
        quantiles = self.forecaster.predict_quantiles(X_trans)

        latency_ms = (time.perf_counter() - start) * 1000.0

        nominal_days = float(shipment_dict.get("nominal_transit_days", 14.0))
        expected_delay_days = max(0.0, quantiles["p50"] - nominal_days)

        return {
            "disruption_probability": round(proba, 4),
            "is_delayed_predicted": is_delayed,
            "risk_tier": risk_tier,
            "mitigation_strategy": mitigation_strategy,
            "lead_time_quantiles_days": quantiles,
            "expected_delay_days": round(expected_delay_days, 2),
            "latency_ms": round(latency_ms, 2)
        }

    def optimize_inventory(
        self,
        mean_lead_time_days: float,
        lead_time_p90_days: float,
        daily_demand_mean: float,
        daily_demand_std: float,
        order_cost_usd: float = 250.0,
        holding_cost_annual_pct: float = 0.20,
        unit_cost_usd: float = 50.0,
        service_level: float = 0.95
    ) -> Dict[str, Any]:
        """Prescriptive inventory optimization for safety stock, ROP, and EOQ."""
        start = time.perf_counter()
        policy = self.optimizer.optimize_inventory_policy(
            mean_lead_time_days=mean_lead_time_days,
            lead_time_p90_days=lead_time_p90_days,
            daily_demand_mean=daily_demand_mean,
            daily_demand_std=daily_demand_std,
            order_cost_usd=order_cost_usd,
            holding_cost_annual_pct=holding_cost_annual_pct,
            unit_cost_usd=unit_cost_usd,
            service_level=service_level
        )
        policy["latency_ms"] = round((time.perf_counter() - start) * 1000.0, 2)
        return policy

    def simulate_shock(
        self,
        base_shipment: Dict[str, Any],
        scenario: str
    ) -> Dict[str, Any]:
        """
        Stress-test supply chain performance under severe macroeconomic or geopolitical shocks:
        - SUEZ_CANAL_BLOCKAGE
        - PORT_LABOR_STRIKE
        - CATEGORY_5_TYPHOON
        - SUPPLIER_BANKRUPTCY
        """
        shocked = base_shipment.copy()

        if scenario == "SUEZ_CANAL_BLOCKAGE":
            shocked["nominal_transit_days"] = shocked.get("nominal_transit_days", 18.0) + 12.0
            shocked["port_congestion_index"] = min(10.0, shocked.get("port_congestion_index", 3.0) + 4.5)
            shocked["bunker_fuel_index"] = shocked.get("bunker_fuel_index", 115.0) + 35.0
        elif scenario == "PORT_LABOR_STRIKE":
            shocked["port_congestion_index"] = 9.8
            shocked["customs_dwell_hours"] = 120.0
        elif scenario == "CATEGORY_5_TYPHOON":
            shocked["weather_risk_score"] = 0.95
            shocked["carrier_rating"] = max(1.0, shocked.get("carrier_rating", 4.0) - 1.5)
        elif scenario == "SUPPLIER_BANKRUPTCY":
            shocked["supplier_reliability"] = 0.15

        base_res = self.predict_shipment(base_shipment)
        shock_res = self.predict_shipment(shocked)

        delta_prob = round(shock_res["disruption_probability"] - base_res["disruption_probability"], 4)
        delta_p50 = round(shock_res["lead_time_quantiles_days"]["p50"] - base_res["lead_time_quantiles_days"]["p50"], 2)

        return {
            "scenario": scenario,
            "baseline_probability": base_res["disruption_probability"],
            "shocked_probability": shock_res["disruption_probability"],
            "delta_disruption_probability": delta_prob,
            "baseline_median_lead_time": base_res["lead_time_quantiles_days"]["p50"],
            "shocked_median_lead_time": shock_res["lead_time_quantiles_days"]["p50"],
            "delta_lead_time_days": delta_p50,
            "shocked_risk_tier": shock_res["risk_tier"],
            "shocked_mitigation": shock_res["mitigation_strategy"]
        }

    def save(self, filepath: str):
        """Bundle and save full engine components."""
        bundle = {
            "feature_engineer": self.feature_engineer,
            "classifier": self.classifier,
            "forecaster": self.forecaster,
            "optimizer": self.optimizer
        }
        joblib.dump(bundle, filepath)

    @classmethod
    def load(cls, filepath: str) -> "OptiChainEngine":
        bundle = joblib.load(filepath)
        return cls(
            feature_engineer=bundle["feature_engineer"],
            classifier=bundle["classifier"],
            forecaster=bundle["forecaster"],
            optimizer=bundle["optimizer"]
        )
