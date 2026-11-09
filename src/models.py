"""
OptiChain AI - Predictive & Prescriptive Machine Learning Models.
Includes:
1. DisruptionClassifier: Supervised Gradient Boosting for binary shipment delay risk.
2. LeadTimeQuantileForecaster: Multi-quantile regressor for p10, p50, p90, p99 lead times.
3. InventoryOptimizer: Multi-echelon Operations Research safety stock and EOQ calculator.
"""

import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingRegressor
from sklearn.base import BaseEstimator, ClassifierMixin


class DisruptionClassifier(BaseEstimator, ClassifierMixin):
    """
    Cost-sensitive gradient boosting classifier for predicting freight SLA breaches.
    """

    def __init__(self, max_iter: int = 150, learning_rate: float = 0.08, max_depth: int = 6):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.clf = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def assess_risk(self, prob: float) -> tuple[str, str]:
        if prob >= 0.70:
            return "CRITICAL", "EXPEDITE_AIR_REROUTE"
        elif prob >= 0.45:
            return "HIGH", "PRIORITIZE_CUSTOMS_BUFFER"
        elif prob >= 0.25:
            return "MEDIUM", "MONITOR_PORT_CONGESTION"
        else:
            return "LOW", "STANDARD_DISPATCH"


class LeadTimeQuantileForecaster:
    """
    Multi-quantile gradient boosting regressors estimating optimistic (p10),
    median (p50), conservative (p90), and tail-risk (p99) actual transit days.
    """

    def __init__(self, n_estimators: int = 60, learning_rate: float = 0.10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models: Dict[str, GradientBoostingRegressor] = {}
        self.quantiles = [0.10, 0.50, 0.90, 0.99]

    def fit(self, X: np.ndarray, y: np.ndarray):
        for q in self.quantiles:
            model_key = f"p{int(q*100)}"
            reg = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=4,
                random_state=42
            )
            reg.fit(X, y)
            self.models[model_key] = reg
        return self

    def predict_quantiles(self, X: np.ndarray) -> Dict[str, float]:
        preds = {}
        for key, model in self.models.items():
            val = float(model.predict(X)[0])
            preds[key] = round(max(1.0, val), 2)
        # Enforce monotonic quantile consistency
        preds["p50"] = max(preds["p10"], preds["p50"])
        preds["p90"] = max(preds["p50"], preds["p90"])
        preds["p99"] = max(preds["p90"], preds["p99"])
        return preds


class InventoryOptimizer:
    """
    Prescriptive Operations Research Engine for Multi-Echelon Inventory Optimization (MEIO).
    Calculates Dynamic Safety Stock, Reorder Points, and Economic Order Quantities.
    """

    @staticmethod
    def calculate_safety_stock(
        mean_lead_time_days: float,
        std_lead_time_days: float,
        daily_demand_mean: float,
        daily_demand_std: float,
        service_level: float = 0.95
    ) -> float:
        """
        Computes dynamic safety stock considering both demand and lead-time volatility:
        SS = Z * sqrt( L_mean * sigma_D^2 + D_mean^2 * sigma_L^2 )
        """
        z_score = float(norm.ppf(service_level))
        variance_term = (
            mean_lead_time_days * (daily_demand_std ** 2) +
            (daily_demand_mean ** 2) * (std_lead_time_days ** 2)
        )
        safety_stock = z_score * math.sqrt(max(0.001, variance_term))
        return round(safety_stock, 2)

    @staticmethod
    def calculate_reorder_point(
        mean_lead_time_days: float,
        daily_demand_mean: float,
        safety_stock: float
    ) -> float:
        """ROP = (Daily Demand * Mean Lead Time) + Safety Stock"""
        rop = (daily_demand_mean * mean_lead_time_days) + safety_stock
        return round(rop, 2)

    @staticmethod
    def calculate_economic_order_quantity(
        annual_demand: float,
        order_cost_usd: float,
        holding_cost_per_unit_year: float
    ) -> float:
        """EOQ = sqrt( (2 * D * S) / H )"""
        if holding_cost_per_unit_year <= 0:
            holding_cost_per_unit_year = 1.0
        eoq = math.sqrt((2.0 * annual_demand * order_cost_usd) / holding_cost_per_unit_year)
        return round(eoq, 2)

    def optimize_inventory_policy(
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
        """
        Synthesize an end-to-end multi-echelon replenishment strategy.
        """
        # Estimate lead time standard deviation from (p90 - p50) / 1.28
        std_lead_time = max(0.5, (lead_time_p90_days - mean_lead_time_days) / 1.282)

        safety_stock = self.calculate_safety_stock(
            mean_lead_time_days=mean_lead_time_days,
            std_lead_time_days=std_lead_time,
            daily_demand_mean=daily_demand_mean,
            daily_demand_std=daily_demand_std,
            service_level=service_level
        )

        rop = self.calculate_reorder_point(
            mean_lead_time_days=mean_lead_time_days,
            daily_demand_mean=daily_demand_mean,
            safety_stock=safety_stock
        )

        annual_demand = daily_demand_mean * 365.0
        holding_cost_unit_year = unit_cost_usd * holding_cost_annual_pct

        eoq = self.calculate_economic_order_quantity(
            annual_demand=annual_demand,
            order_cost_usd=order_cost_usd,
            holding_cost_per_unit_year=holding_cost_unit_year
        )

        annual_orders = round(annual_demand / max(1.0, eoq), 1)
        annual_holding_cost = round((eoq / 2.0 + safety_stock) * holding_cost_unit_year, 2)
        annual_ordering_cost = round(annual_orders * order_cost_usd, 2)
        total_annual_cost = round(annual_holding_cost + annual_ordering_cost, 2)

        return {
            "service_level_target_pct": round(service_level * 100, 1),
            "safety_stock_units": safety_stock,
            "reorder_point_units": rop,
            "economic_order_quantity_units": eoq,
            "orders_per_year": annual_orders,
            "annual_holding_cost_usd": annual_holding_cost,
            "annual_ordering_cost_usd": annual_ordering_cost,
            "total_inventory_cost_usd": total_annual_cost,
            "lead_time_volatility_days": round(std_lead_time, 2)
        }
