"""
OptiChain AI - Enterprise Standard Library Test Suite.
Built on Python's native unittest framework and FastAPI TestClient.
Executes out of the box with zero external test runners required.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient

from src.data_generator import generate_supply_chain_dataset
from src.features import LogisticsFeatureEngineer
from src.models import DisruptionClassifier, LeadTimeQuantileForecaster, InventoryOptimizer
from src.engine import OptiChainEngine
from api.main import app, get_engine


class TestOptiChainSuite(unittest.TestCase):
    """
    Comprehensive verification of OptiChain AI logistics models, inventory logic, and REST microservice endpoints.
    """

    @classmethod
    def setUpClass(cls):
        model_path = os.path.join(BASE_DIR, "artifacts", "optichain_engine.joblib")
        if os.path.exists(model_path):
            cls.engine = OptiChainEngine.load(model_path)
        else:
            cls.engine = OptiChainEngine()
        cls.client = TestClient(app)

    def test_01_data_generator_contract(self):
        """Verify synthetic benchmark dataset conforms to supply chain domain schema."""
        df = generate_supply_chain_dataset(n_samples=250, random_state=42)
        self.assertEqual(len(df), 250)
        self.assertIn("nominal_transit_days", df.columns)
        self.assertIn("actual_transit_days", df.columns)
        self.assertIn("is_delayed", df.columns)
        self.assertIn("delay_severity", df.columns)
        self.assertTrue((df["actual_transit_days"] > 0).all())
        self.assertTrue((df["is_delayed"].isin([0, 1])).all())

    def test_02_feature_engineering_pipeline(self):
        """Verify feature transformations and engineered interaction terms."""
        df = generate_supply_chain_dataset(n_samples=100, random_state=42)
        feature_cols = [c for c in df.columns if c not in ["shipment_id", "actual_transit_days", "delay_days", "is_delayed", "delay_severity"]]
        fe = LogisticsFeatureEngineer()
        X_trans = fe.fit_transform(df[feature_cols])
        self.assertEqual(X_trans.shape[0], 100)
        self.assertGreater(X_trans.shape[1], 15)
        self.assertFalse(np.isnan(X_trans).any())

    def test_03_disruption_classifier_prediction(self):
        """Verify classification probabilities and risk tier assignment."""
        proba_low = 0.12
        tier_low, strat_low = self.engine.classifier.assess_risk(proba_low)
        self.assertEqual(tier_low, "LOW")
        self.assertEqual(strat_low, "STANDARD_DISPATCH")

        proba_high = 0.78
        tier_high, strat_high = self.engine.classifier.assess_risk(proba_high)
        self.assertEqual(tier_high, "CRITICAL")
        self.assertEqual(strat_high, "EXPEDITE_AIR_REROUTE")

    def test_04_lead_time_quantiles_monotonicity(self):
        """Verify quantile forecasts maintain physical monotonicity: p10 <= p50 <= p90 <= p99."""
        sample_shipment = {
            "origin_port": "Shanghai (CNSHA)",
            "destination_port": "Long Beach (USLGB)",
            "transport_mode": "Ocean Freight",
            "nominal_transit_days": 18.0,
            "carrier_rating": 4.1,
            "supplier_reliability": 0.90,
            "port_congestion_index": 3.2,
            "weather_risk_score": 0.15,
            "customs_inspection": 0,
            "customs_dwell_hours": 4.0,
            "cargo_value_usd": 180000.0,
            "is_perishable": 0,
            "container_teu": 2,
            "bunker_fuel_index": 115.0
        }
        res = self.engine.predict_shipment(sample_shipment)
        q = res["lead_time_quantiles_days"]
        self.assertLessEqual(q["p10"], q["p50"])
        self.assertLessEqual(q["p50"], q["p90"])
        self.assertLessEqual(q["p90"], q["p99"])

    def test_05_safety_stock_scaling(self):
        """Verify that dynamic safety stock strictly scales with higher target service levels."""
        opt = InventoryOptimizer()
        ss_90 = opt.calculate_safety_stock(18.0, 3.0, 100.0, 20.0, service_level=0.90)
        ss_95 = opt.calculate_safety_stock(18.0, 3.0, 100.0, 20.0, service_level=0.95)
        ss_99 = opt.calculate_safety_stock(18.0, 3.0, 100.0, 20.0, service_level=0.99)
        self.assertLess(ss_90, ss_95)
        self.assertLess(ss_95, ss_99)

    def test_06_eoq_economic_formula(self):
        """Verify that Economic Order Quantity conforms to Wilson's square root law."""
        opt = InventoryOptimizer()
        eoq_base = opt.calculate_economic_order_quantity(annual_demand=10000, order_cost_usd=200, holding_cost_per_unit_year=10)
        # Quadrupling demand should double EOQ
        eoq_quad = opt.calculate_economic_order_quantity(annual_demand=40000, order_cost_usd=200, holding_cost_per_unit_year=10)
        self.assertAlmostEqual(eoq_quad, eoq_base * 2.0, delta=1.0)

    def test_07_engine_predict_shipment(self):
        """Verify end-to-end single shipment prediction."""
        shipment = {
            "origin_port": "Rotterdam (NLRTM)",
            "destination_port": "New York (USNYC)",
            "transport_mode": "Ocean Freight",
            "nominal_transit_days": 12.0,
            "carrier_rating": 3.9,
            "supplier_reliability": 0.88,
            "port_congestion_index": 4.0,
            "weather_risk_score": 0.30,
            "customs_inspection": 1,
            "customs_dwell_hours": 18.0,
            "cargo_value_usd": 250000.0,
            "is_perishable": 1,
            "container_teu": 4,
            "bunker_fuel_index": 122.0
        }
        res = self.engine.predict_shipment(shipment)
        self.assertIn("disruption_probability", res)
        self.assertIn("risk_tier", res)
        self.assertIn("mitigation_strategy", res)
        self.assertGreater(res["disruption_probability"], 0.0)

    def test_08_engine_shock_simulation(self):
        """Verify that severe supply shocks increase delay probabilities and lead times."""
        base_shipment = {
            "origin_port": "Shanghai (CNSHA)",
            "destination_port": "Long Beach (USLGB)",
            "transport_mode": "Ocean Freight",
            "nominal_transit_days": 18.0,
            "carrier_rating": 4.5,
            "supplier_reliability": 0.95,
            "port_congestion_index": 2.0,
            "weather_risk_score": 0.10,
            "customs_inspection": 0,
            "customs_dwell_hours": 2.0,
            "cargo_value_usd": 120000.0,
            "is_perishable": 0,
            "container_teu": 2,
            "bunker_fuel_index": 110.0
        }
        res = self.engine.simulate_shock(base_shipment, "SUEZ_CANAL_BLOCKAGE")
        self.assertGreater(res["shocked_probability"], res["baseline_probability"])
        self.assertGreater(res["shocked_median_lead_time"], res["baseline_median_lead_time"])

    def test_09_api_health_endpoint(self):
        """Verify REST API /health endpoint returns HTTP 200 and operational capabilities."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn(data["status"], ["healthy", "degraded"])
        self.assertIn("capabilities", data)

    def test_10_api_predict_delay_endpoint(self):
        """Verify REST API /v1/predict/delay endpoint."""
        payload = {
            "origin_port": "Hamburg (DEHAM)",
            "destination_port": "New York (USNYC)",
            "transport_mode": "Ocean Freight",
            "nominal_transit_days": 14.0,
            "carrier_rating": 4.0,
            "supplier_reliability": 0.91,
            "port_congestion_index": 3.0,
            "weather_risk_score": 0.20,
            "customs_inspection": 0,
            "customs_dwell_hours": 4.0,
            "cargo_value_usd": 160000.0,
            "is_perishable": 0,
            "container_teu": 2,
            "bunker_fuel_index": 118.0
        }
        resp = self.client.post("/v1/predict/delay", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("disruption_probability", data)
        self.assertIn("lead_time_quantiles_days", data)

    def test_11_api_optimize_inventory_endpoint(self):
        """Verify REST API /v1/optimize/inventory endpoint."""
        payload = {
            "mean_lead_time_days": 18.0,
            "lead_time_p90_days": 23.5,
            "daily_demand_mean": 95.0,
            "daily_demand_std": 18.0,
            "order_cost_usd": 300.0,
            "holding_cost_annual_pct": 0.20,
            "unit_cost_usd": 40.0,
            "service_level": 0.95
        }
        resp = self.client.post("/v1/optimize/inventory", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["safety_stock_units"], 0)
        self.assertGreater(data["economic_order_quantity_units"], 0)
        self.assertGreater(data["reorder_point_units"], 0)

    def test_12_api_simulate_shock_endpoint(self):
        """Verify REST API /v1/simulate/shock endpoint."""
        payload = {
            "base_shipment": {
                "origin_port": "Singapore (SGSIN)",
                "destination_port": "Rotterdam (NLRTM)",
                "transport_mode": "Ocean Freight",
                "nominal_transit_days": 22.0,
                "carrier_rating": 4.3,
                "supplier_reliability": 0.92,
                "port_congestion_index": 3.1,
                "weather_risk_score": 0.15,
                "customs_inspection": 0,
                "customs_dwell_hours": 3.0,
                "cargo_value_usd": 210000.0,
                "is_perishable": 0,
                "container_teu": 4,
                "bunker_fuel_index": 116.0
            },
            "scenario": "PORT_LABOR_STRIKE"
        }
        resp = self.client.post("/v1/simulate/shock", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["scenario"], "PORT_LABOR_STRIKE")
        self.assertGreaterEqual(data["shocked_probability"], data["baseline_probability"])

    def test_13_api_batch_predict_endpoint(self):
        """Verify REST API /v1/batch/predict endpoint."""
        payload = [
            {
                "origin_port": "Busan (KRPUS)",
                "destination_port": "Long Beach (USLGB)",
                "transport_mode": "Ocean Freight",
                "nominal_transit_days": 16.0,
                "carrier_rating": 4.2,
                "supplier_reliability": 0.90,
                "port_congestion_index": 3.0,
                "weather_risk_score": 0.20,
                "customs_inspection": 0,
                "customs_dwell_hours": 4.0,
                "cargo_value_usd": 150000.0,
                "is_perishable": 0,
                "container_teu": 2,
                "bunker_fuel_index": 115.0
            }
        ]
        resp = self.client.post("/v1/batch/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["batch_size"], 1)

    def test_14_api_metrics_endpoint(self):
        """Verify REST API /v1/metrics endpoint returns offline benchmark metrics."""
        resp = self.client.get("/v1/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("classification_metrics", data)

    def test_15_model_persistence(self):
        """Verify engine serialization and deserialization roundtrip."""
        temp_path = os.path.join(BASE_DIR, "artifacts", "temp_engine_test.joblib")
        try:
            self.engine.save(temp_path)
            loaded = OptiChainEngine.load(temp_path)
            self.assertIsNotNone(loaded.classifier)
            self.assertIsNotNone(loaded.forecaster)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
