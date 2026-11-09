"""
OptiChain AI - Pydantic Request & Response Schemas.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ShipmentPredictRequest(BaseModel):
    origin_port: str = Field("Shanghai (CNSHA)", description="Port of departure")
    destination_port: str = Field("Long Beach (USLGB)", description="Port of arrival")
    transport_mode: str = Field("Ocean Freight", description="Freight transport mode")
    nominal_transit_days: float = Field(18.0, ge=0.5, le=90.0, description="Scheduled transit duration in days")
    carrier_rating: float = Field(4.2, ge=1.0, le=5.0, description="Carrier on-time rating")
    supplier_reliability: float = Field(0.92, ge=0.0, le=1.0, description="Manufacturing supplier reliability score")
    port_congestion_index: float = Field(3.5, ge=1.0, le=10.0, description="Port dwell congestion index")
    weather_risk_score: float = Field(0.20, ge=0.0, le=1.0, description="Route weather disruption probability")
    customs_inspection: int = Field(0, ge=0, le=1, description="Customs intensive examination flag")
    customs_dwell_hours: float = Field(4.0, ge=0.0, le=168.0, description="Customs clearance dwell time in hours")
    cargo_value_usd: float = Field(150000.0, ge=100.0, description="Shipment commercial invoice value in USD")
    is_perishable: int = Field(0, ge=0, le=1, description="Cold-chain or perishable flag")
    container_teu: int = Field(2, ge=1, le=50, description="Container volume in TEU units")
    bunker_fuel_index: float = Field(115.0, ge=50.0, le=250.0, description="Bunker fuel price benchmark index")

    model_config = {
        "json_schema_extra": {
            "example": {
                "origin_port": "Shanghai (CNSHA)",
                "destination_port": "Long Beach (USLGB)",
                "transport_mode": "Ocean Freight",
                "nominal_transit_days": 18.0,
                "carrier_rating": 3.8,
                "supplier_reliability": 0.85,
                "port_congestion_index": 4.5,
                "weather_risk_score": 0.25,
                "customs_inspection": 0,
                "customs_dwell_hours": 6.0,
                "cargo_value_usd": 220000.0,
                "is_perishable": 0,
                "container_teu": 4,
                "bunker_fuel_index": 120.0
            }
        }
    }


class ShipmentPredictResponse(BaseModel):
    disruption_probability: float
    is_delayed_predicted: bool
    risk_tier: str
    mitigation_strategy: str
    lead_time_quantiles_days: Dict[str, float]
    expected_delay_days: float
    latency_ms: float


class InventoryOptimizeRequest(BaseModel):
    mean_lead_time_days: float = Field(..., ge=0.5, description="Expected average transit lead time in days")
    lead_time_p90_days: float = Field(..., ge=0.5, description="90th percentile conservative lead time in days")
    daily_demand_mean: float = Field(..., ge=1.0, description="Mean daily demand in units")
    daily_demand_std: float = Field(..., ge=0.1, description="Standard deviation of daily demand in units")
    order_cost_usd: Optional[float] = Field(250.0, ge=1.0, description="Fixed order placement setup cost in USD")
    holding_cost_annual_pct: Optional[float] = Field(0.20, ge=0.01, le=1.0, description="Annual inventory holding cost fraction")
    unit_cost_usd: Optional[float] = Field(50.0, ge=0.1, description="Unit cost of product in USD")
    service_level: Optional[float] = Field(0.95, ge=0.75, le=0.999, description="Target customer order fulfillment SLA")

    model_config = {
        "json_schema_extra": {
            "example": {
                "mean_lead_time_days": 18.5,
                "lead_time_p90_days": 23.0,
                "daily_demand_mean": 80.0,
                "daily_demand_std": 15.0,
                "order_cost_usd": 300.0,
                "holding_cost_annual_pct": 0.22,
                "unit_cost_usd": 45.0,
                "service_level": 0.95
            }
        }
    }


class InventoryOptimizeResponse(BaseModel):
    service_level_target_pct: float
    safety_stock_units: float
    reorder_point_units: float
    economic_order_quantity_units: float
    orders_per_year: float
    annual_holding_cost_usd: float
    annual_ordering_cost_usd: float
    total_inventory_cost_usd: float
    lead_time_volatility_days: float
    latency_ms: float


class ShockSimulateRequest(BaseModel):
    base_shipment: Dict[str, Any]
    scenario: str = Field(..., description="Shock scenario name (e.g. SUEZ_CANAL_BLOCKAGE, PORT_LABOR_STRIKE, CATEGORY_5_TYPHOON, SUPPLIER_BANKRUPTCY)")


class ShockSimulateResponse(BaseModel):
    scenario: str
    baseline_probability: float
    shocked_probability: float
    delta_disruption_probability: float
    baseline_median_lead_time: float
    shocked_median_lead_time: float
    delta_lead_time_days: float
    shocked_risk_tier: str
    shocked_mitigation: str
