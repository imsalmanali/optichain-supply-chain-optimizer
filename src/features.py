"""
OptiChain AI - Domain Feature Engineering Pipeline.
Constructs multi-modal logistics interactions, vulnerability indices,
dwell-to-transit ratios, and normalized tensor representations.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


class LogisticsFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms raw shipment telemetry into domain-specific logistics features.
    """

    CATEGORICAL_COLS = ["transport_mode", "origin_port", "destination_port"]
    NUMERICAL_COLS = [
        "nominal_transit_days", "carrier_rating", "supplier_reliability",
        "port_congestion_index", "weather_risk_score", "customs_dwell_hours",
        "cargo_value_usd", "is_perishable", "container_teu", "bunker_fuel_index",
        # Engineered interaction terms:
        "congestion_dwell_interaction", "weather_carrier_vulnerability",
        "supplier_risk_factor", "dwell_to_transit_ratio", "value_per_teu"
    ]

    def __init__(self):
        self.preprocessor: ColumnTransformer = None
        self.feature_names: List[str] = []

    def _engineer_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()

        # 1. Congestion & Customs bottleneck coupling
        df_out["congestion_dwell_interaction"] = (
            df_out["port_congestion_index"] * (df_out["customs_dwell_hours"] / 24.0)
        )

        # 2. Weather shock multiplied by carrier unreliability
        df_out["weather_carrier_vulnerability"] = (
            df_out["weather_risk_score"] * (5.0 - df_out["carrier_rating"])
        )

        # 3. Direct supplier fragility index
        df_out["supplier_risk_factor"] = 1.0 - df_out["supplier_reliability"]

        # 4. Customs dwell fraction of total expected transit
        df_out["dwell_to_transit_ratio"] = (
            (df_out["customs_dwell_hours"] / 24.0) / np.maximum(1.0, df_out["nominal_transit_days"])
        )

        # 5. Financial cargo exposure per TEU container
        df_out["value_per_teu"] = (
            df_out["cargo_value_usd"] / np.maximum(1, df_out["container_teu"])
        )

        return df_out

    def fit(self, X: pd.DataFrame, y=None):
        X_eng = self._engineer_interactions(X)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.NUMERICAL_COLS),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.CATEGORICAL_COLS)
            ]
        )
        self.preprocessor.fit(X_eng)

        # Extract generated feature names
        cat_encoder = self.preprocessor.named_transformers_["cat"]
        cat_names = list(cat_encoder.get_feature_names_out(self.CATEGORICAL_COLS))
        self.feature_names = self.NUMERICAL_COLS + cat_names
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_eng = self._engineer_interactions(X)
        return self.preprocessor.transform(X_eng)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)
