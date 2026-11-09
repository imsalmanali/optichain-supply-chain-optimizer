"""
OptiChain AI - Enterprise Production FastAPI Logistics Microservice.
High-throughput REST API for freight delay forecasting, quantile lead-time prediction,
and multi-echelon inventory replenishment optimization.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.engine import OptiChainEngine
from api.schemas import (
    ShipmentPredictRequest, ShipmentPredictResponse,
    InventoryOptimizeRequest, InventoryOptimizeResponse,
    ShockSimulateRequest, ShockSimulateResponse
)


app = FastAPI(
    title="OptiChain AI — Enterprise Supply Chain Intelligence Gateway",
    description=(
        "Production-grade, sub-100ms Supply Chain Resilience & Inventory Optimization microservice. "
        "Provides multi-modal shipment delay prediction, probabilistic lead-time quantile regression, "
        "and dynamic safety stock & EOQ replenishment calculation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: OptiChainEngine = None
_startup_time = time.time()


def get_engine() -> OptiChainEngine:
    global _engine
    if _engine is None:
        model_path = os.path.join(BASE_DIR, "artifacts", "optichain_engine.joblib")
        if os.path.exists(model_path):
            _engine = OptiChainEngine.load(model_path)
        else:
            _engine = OptiChainEngine()
    return _engine


@app.on_event("startup")
def startup_event():
    get_engine()


@app.get("/health", tags=["System Diagnostics"])
def health_check() -> Dict[str, Any]:
    """Inspect microservice uptime, model loaded state, and operational capabilities."""
    uptime_seconds = round(time.time() - _startup_time, 2)
    model_loaded = (
        _engine is not None and
        _engine.classifier is not None and
        _engine.forecaster is not None
    )
    return {
        "status": "healthy" if model_loaded else "degraded",
        "service": "OptiChain AI Logistics Microservice",
        "version": "1.0.0",
        "uptime_seconds": uptime_seconds,
        "model_loaded": model_loaded,
        "capabilities": [
            "FREIGHT_DELAY_CLASSIFICATION",
            "QUANTILE_LEAD_TIME_REGRESSION",
            "DYNAMIC_SAFETY_STOCK_OPTIMIZATION",
            "REORDER_POINT_POLICY",
            "ECONOMIC_ORDER_QUANTITY_EOQ",
            "GEOPOLITICAL_SHOCK_SIMULATION"
        ]
    }


@app.post("/v1/predict/delay", response_model=ShipmentPredictResponse, tags=["Predictive Logistics"])
def predict_shipment_delay(payload: ShipmentPredictRequest):
    """
    Predict SLA delay risk, disruption probability, and lead-time quantiles (p10, p50, p90, p99) for a single shipment.
    """
    engine = get_engine()
    try:
        data_dict = payload.model_dump()
        return engine.predict_shipment(data_dict)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Shipment delay prediction failed: {str(e)}"
        )


@app.post("/v1/optimize/inventory", response_model=InventoryOptimizeResponse, tags=["Prescriptive Inventory"])
def optimize_inventory_endpoint(payload: InventoryOptimizeRequest):
    """
    Calculate dynamic safety stock, reorder point (ROP), and economic order quantity (EOQ) based on lead-time volatility.
    """
    engine = get_engine()
    try:
        return engine.optimize_inventory(
            mean_lead_time_days=payload.mean_lead_time_days,
            lead_time_p90_days=payload.lead_time_p90_days,
            daily_demand_mean=payload.daily_demand_mean,
            daily_demand_std=payload.daily_demand_std,
            order_cost_usd=payload.order_cost_usd,
            holding_cost_annual_pct=payload.holding_cost_annual_pct,
            unit_cost_usd=payload.unit_cost_usd,
            service_level=payload.service_level
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inventory replenishment optimization failed: {str(e)}"
        )


@app.post("/v1/simulate/shock", response_model=ShockSimulateResponse, tags=["Disruption Simulation"])
def simulate_shock_endpoint(payload: ShockSimulateRequest):
    """
    Simulate macroeconomic or geopolitical supply chain shocks (e.g. SUEZ_CANAL_BLOCKAGE, PORT_LABOR_STRIKE, TYPHOON).
    """
    engine = get_engine()
    try:
        return engine.simulate_shock(
            base_shipment=payload.base_shipment,
            scenario=payload.scenario
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Supply chain shock simulation failed: {str(e)}"
        )


@app.post("/v1/batch/predict", tags=["Predictive Logistics"])
def batch_predict_shipments(shipments: List[ShipmentPredictRequest]) -> Dict[str, Any]:
    """
    High-throughput batch prediction for an entire freight manifest or portfolio of active transit shipments.
    """
    engine = get_engine()
    try:
        results = []
        for s in shipments:
            pred = engine.predict_shipment(s.model_dump())
            results.append(pred)
        return {
            "batch_size": len(shipments),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/v1/metrics", tags=["System Diagnostics"])
def get_benchmark_metrics() -> Dict[str, Any]:
    """Retrieve certified offline benchmark metrics and evaluation profiles."""
    metrics_path = os.path.join(BASE_DIR, "artifacts", "benchmark_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"message": "Benchmark metrics have not yet been generated. Run src/train.py first."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
