"""
OptiChain AI - Automated Training, Benchmarking & Artifact Generation Pipeline.
Trains DisruptionClassifier and LeadTimeQuantileForecaster, evaluates precision metrics,
renders diagnostic plots, and saves production artifacts.
"""

import os
import sys

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, mean_absolute_error, root_mean_squared_error
)

from src.data_generator import generate_supply_chain_dataset
from src.features import LogisticsFeatureEngineer
from src.models import DisruptionClassifier, LeadTimeQuantileForecaster, InventoryOptimizer
from src.engine import OptiChainEngine


def run_training_pipeline():
    print("=" * 70)
    print("[OptiChain AI] Initiating Global Supply Chain Logistics Training Pipeline")
    print("=" * 70)

    data_dir = os.path.join(BASE_DIR, "data")
    artifacts_dir = os.path.join(BASE_DIR, "artifacts")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Dataset Generation
    print("[*] Generating 3,500 global freight shipment records across 4 transport modes...")
    df = generate_supply_chain_dataset(n_samples=3500, random_state=42)
    csv_path = os.path.join(data_dir, "global_supply_chain_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Benchmark dataset saved to {csv_path} ({len(df)} samples)")
    print("   Delay Severity Breakdown:")
    for sev, cnt in df["delay_severity"].value_counts().items():
        print(f"   - {sev:<24}: {cnt} shipments")

    # 2. Train/Test Split
    feature_cols = [c for c in df.columns if c not in ["shipment_id", "actual_transit_days", "delay_days", "is_delayed", "delay_severity"]]
    X_raw = df[feature_cols]
    y_class = np.array(df["is_delayed"].tolist(), dtype=int)
    y_reg = np.array(df["actual_transit_days"].tolist(), dtype=float)

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=y_class)

    X_train_raw = X_raw.iloc[train_idx].reset_index(drop=True)
    X_test_raw = X_raw.iloc[test_idx].reset_index(drop=True)
    y_class_train, y_class_test = y_class[train_idx], y_class[test_idx]
    y_reg_train, y_reg_test = y_reg[train_idx], y_reg[test_idx]

    print(f"\n[*] Partitions: {len(train_idx)} training shipments, {len(test_idx)} testing shipments.")

    # 3. Fit Feature Engineering Pipeline
    print("[*] Fitting domain logistics feature preprocessor...")
    fe = LogisticsFeatureEngineer()
    X_train_trans = fe.fit_transform(X_train_raw)
    X_test_trans = fe.transform(X_test_raw)
    print(f"[OK] Feature preprocessor fitted with {X_train_trans.shape[1]} engineered dimensions.")

    # 4. Fit Disruption Classifier
    print("[*] Training Cost-Sensitive Gradient Boosting Disruption Classifier...")
    t0 = time.time()
    clf = DisruptionClassifier(max_iter=150, learning_rate=0.08)
    clf.fit(X_train_trans, y_class_train)
    clf_time = time.time() - t0
    print(f"[OK] Disruption classifier trained in {clf_time:.2f} seconds.")

    # 5. Fit Quantile Forecasters
    print("[*] Training Multi-Quantile Lead Time Regressors (p10, p50, p90, p99)...")
    t0 = time.time()
    forecaster = LeadTimeQuantileForecaster(n_estimators=70, learning_rate=0.10)
    forecaster.fit(X_train_trans, y_reg_train)
    reg_time = time.time() - t0
    print(f"[OK] Quantile regressors trained in {reg_time:.2f} seconds.")

    # 6. Evaluation & Benchmarking
    print("[*] Evaluating predictive performance on held-out test partition...")
    y_pred_probs = clf.predict_proba(X_test_trans)[:, 1]
    y_pred_class = (y_pred_probs >= 0.50).astype(int)

    acc = float(accuracy_score(y_class_test, y_pred_class))
    prec = float(precision_score(y_class_test, y_pred_class, zero_division=0))
    rec = float(recall_score(y_class_test, y_pred_class, zero_division=0))
    f1 = float(f1_score(y_class_test, y_pred_class, zero_division=0))
    roc_auc = float(roc_auc_score(y_class_test, y_pred_probs))
    pr_auc = float(average_precision_score(y_class_test, y_pred_probs))
    cm = confusion_matrix(y_class_test, y_pred_class).tolist()

    # Lead time median error
    p50_preds = forecaster.models["p50"].predict(X_test_trans)
    mae = float(mean_absolute_error(y_reg_test, p50_preds))
    rmse = float(root_mean_squared_error(y_reg_test, p50_preds))

    # Latency benchmark
    latencies = []
    engine = OptiChainEngine(feature_engineer=fe, classifier=clf, forecaster=forecaster)
    for i in range(min(300, len(test_idx))):
        record = X_test_raw.iloc[i].to_dict()
        res = engine.predict_shipment(record)
        latencies.append(res["latency_ms"])

    avg_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))

    print("\n[+] Benchmark Results:")
    print(f"   * Disruption Accuracy:    {acc * 100:.2f}%")
    print(f"   * Disruption Precision:   {prec * 100:.2f}%")
    print(f"   * Disruption Recall:      {rec * 100:.2f}%")
    print(f"   * Disruption F1-Score:    {f1 * 100:.2f}%")
    print(f"   * Disruption ROC-AUC:     {roc_auc:.4f}")
    print(f"   * Disruption PR-AUC:      {pr_auc:.4f}")
    print(f"   * Lead Time Median MAE:   {mae:.2f} days")
    print(f"   * Lead Time Median RMSE:  {rmse:.2f} days")
    print(f"   * Prediction Latency:     Avg={avg_lat:.2f}ms | p50={p50_lat:.2f}ms | p95={p95_lat:.2f}ms | p99={p99_lat:.2f}ms")

    # 7. Serialize Artifacts
    model_save_path = os.path.join(artifacts_dir, "optichain_engine.joblib")
    engine.save(model_save_path)
    print(f"\n[OK] Model engine serialized to: {model_save_path}")

    metrics_dict = {
        "model_name": "OptiChain-SupplyChainResilience-v1",
        "dataset_samples": len(df),
        "test_samples": len(test_idx),
        "classification_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": cm
        },
        "regression_lead_time_metrics": {
            "median_mae_days": round(mae, 3),
            "median_rmse_days": round(rmse, 3)
        },
        "latency_benchmarks_ms": {
            "average": round(avg_lat, 3),
            "median_p50": round(p50_lat, 3),
            "percentile_p95": round(p95_lat, 3),
            "percentile_p99": round(p99_lat, 3)
        },
        "severity_distribution": df["delay_severity"].value_counts().to_dict()
    }

    metrics_save_path = os.path.join(artifacts_dir, "benchmark_metrics.json")
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"[OK] Metrics saved to: {metrics_save_path}")

    # 8. Render Visual Diagnostics
    print("\n[*] Rendering diagnostic plots and figures...")
    sns.set_theme(style="whitegrid", palette="muted")

    # Figure 1: Model Diagnostics (4 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Subplot 1: Confusion Matrix
    cm_arr = np.array(cm)
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0, 0],
        xticklabels=["Predicted On-Time", "Predicted Delayed"],
        yticklabels=["Actual On-Time", "Actual Delayed"]
    )
    axes[0, 0].set_title(f"Disruption Confusion Matrix (Acc: {acc*100:.1f}%)", fontsize=12, fontweight="bold")

    # Subplot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_class_test, y_pred_probs)
    axes[0, 1].plot(fpr, tpr, color="#009688", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    axes[0, 1].plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel("False Positive Rate (FPR)")
    axes[0, 1].set_ylabel("True Positive Rate (TPR / Recall)")
    axes[0, 1].set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    axes[0, 1].legend(loc="lower right")

    # Subplot 3: Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_class_test, y_pred_probs)
    axes[1, 0].plot(recall_vals, precision_vals, color="#E91E63", lw=2.5, label=f"PR Curve (PR-AUC = {pr_auc:.4f})")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    axes[1, 0].legend(loc="lower left")

    # Subplot 4: Delay Severity Distribution
    sev_counts = df["delay_severity"].value_counts()
    axes[1, 1].bar(sev_counts.index, sev_counts.values, color=["#4CAF50", "#FFC107", "#FF9800", "#F44336"])
    axes[1, 1].set_ylabel("Shipment Volume")
    axes[1, 1].set_title("Global Logistics Delay Severity Distribution", fontsize=12, fontweight="bold")
    axes[1, 1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    diag_path = os.path.join(artifacts_dir, "model_diagnostics.png")
    fig.savefig(diag_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Model diagnostics saved to: {diag_path}")

    # Figure 2: Lead Time Quantile Fan & Safety Stock Curve
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Fan chart for 20 sample test shipments
    sample_sub = X_test_trans[:20]
    p10_s = forecaster.models["p10"].predict(sample_sub)
    p50_s = forecaster.models["p50"].predict(sample_sub)
    p90_s = forecaster.models["p90"].predict(sample_sub)
    p99_s = forecaster.models["p99"].predict(sample_sub)
    actual_s = y_reg_test[:20]

    x_axis = np.arange(20)
    axes2[0].fill_between(x_axis, p10_s, p99_s, color="#B3E5FC", alpha=0.6, label="Tail Risk Interval [p10 - p99]")
    axes2[0].fill_between(x_axis, p50_s, p90_s, color="#4FC3F7", alpha=0.8, label="Expected Interval [p50 - p90]")
    axes2[0].plot(x_axis, p50_s, color="#0288D1", lw=2, marker="o", label="Median Transit Forecast (p50)")
    axes2[0].scatter(x_axis, actual_s, color="red", zorder=5, label="Actual Transit Days")
    axes2[0].set_xlabel("Sample Shipment Index")
    axes2[0].set_ylabel("Transit Time (Days)")
    axes2[0].set_title("Probabilistic Lead Time Quantile Forecast Fan", fontsize=11, fontweight="bold")
    axes2[0].legend(loc="upper left", fontsize=8)

    # Safety Stock Trade-Off vs Service Level Curve
    service_levels = np.linspace(0.80, 0.999, 50)
    opt = InventoryOptimizer()
    safety_stocks = [
        opt.calculate_safety_stock(
            mean_lead_time_days=18.0,
            std_lead_time_days=3.5,
            daily_demand_mean=100.0,
            daily_demand_std=20.0,
            service_level=sl
        )
        for sl in service_levels
    ]
    axes2[1].plot(service_levels * 100, safety_stocks, color="#7B1FA2", lw=2.5)
    axes2[1].axvline(95.0, color="gray", linestyle="--", label="Standard 95% SLA Target")
    axes2[1].axvline(99.0, color="orange", linestyle=":", label="Mission-Critical 99% SLA Target")
    axes2[1].set_xlabel("Service Level Target (%)")
    axes2[1].set_ylabel("Required Safety Stock (Units)")
    axes2[1].set_title("Safety Stock Buffer vs Target Service Level SLA", fontsize=11, fontweight="bold")
    axes2[1].legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    quant_path = os.path.join(artifacts_dir, "lead_time_quantiles.png")
    fig2.savefig(quant_path, dpi=200)
    plt.close(fig2)
    print(f"[OK] Lead time quantile chart saved to: {quant_path}")

    print("\n[SUCCESS] OptiChain training pipeline successfully completed!")
    return metrics_dict


if __name__ == "__main__":
    run_training_pipeline()
