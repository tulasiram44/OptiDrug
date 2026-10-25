"""
OptiDrug - Automated Test Suite
Verifies data generation, preprocessing, Random Forest training & metrics,
multi-objective optimization, and chemical utility calculations.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
from data.sample_data import generate_target_dataset, TARGET_CATALOG
from pipeline.preprocessing import DataPreprocessor
from pipeline.model import BioactivityPredictor
from pipeline.optimizer import MultiObjectiveOptimizer
from utils.chemical import assess_lipinski_rule_of_5, assess_veber_rules, generate_compound_svg
from data.ncbi_client import fetch_gene_summary

def test_dataset_generation():
    df = generate_target_dataset("EGFR", n_samples=250, random_state=42)
    assert len(df) > 250  # includes injected duplicates
    assert "compound_id" in df.columns
    assert "molecular_weight" in df.columns
    assert "activity_score" in df.columns
    assert "pic50" in df.columns
    assert df["molecular_weight"].min() >= 300
    assert df["molecular_weight"].max() <= 600

def test_preprocessing_pipeline():
    df = generate_target_dataset("EGFR", n_samples=250, random_state=42)
    prep = DataPreprocessor(random_state=42)
    X_train_norm, X_test_norm, y_train, y_test, cleaned_df = prep.fit_transform(df)
    
    # Audit checks
    assert prep.audit_log["duplicate_rows_removed"] > 0
    assert len(cleaned_df) == 250
    assert len(X_train_norm) == 200
    assert len(X_test_norm) == 50
    
    # Check normalized bounds [0, 1]
    assert X_train_norm.min().min() >= -1e-6
    assert X_train_norm.max().max() <= 1.0 + 1e-6

def test_model_training_and_metrics():
    df = generate_target_dataset("EGFR", n_samples=250, random_state=42)
    prep = DataPreprocessor(random_state=42)
    X_train_norm, X_test_norm, y_train, y_test, _ = prep.fit_transform(df)
    
    rf = BioactivityPredictor(n_estimators=100, max_depth=6, random_state=42)
    results = rf.train_and_evaluate(X_train_norm, X_test_norm, y_train, y_test)
    
    metrics = results["metrics"]
    # Check benchmark alignment
    assert 0.52 <= metrics["test_r2"] <= 0.65
    assert 0.060 <= metrics["test_mae"] <= 0.085
    assert len(results["feature_importance"]) > 0

def test_multi_objective_optimization():
    df = generate_target_dataset("EGFR", n_samples=250, random_state=42)
    prep = DataPreprocessor(random_state=42)
    X_train_norm, X_test_norm, y_train, y_test, cleaned_df = prep.fit_transform(df)
    rf = BioactivityPredictor(n_estimators=100, max_depth=6, random_state=42)
    rf.train_and_evaluate(X_train_norm, X_test_norm, y_train, y_test)
    
    X_all_norm = prep.transform_candidates(cleaned_df)
    preds = rf.predict(X_all_norm)
    
    optimizer = MultiObjectiveOptimizer()
    optimized = optimizer.optimize_candidates(cleaned_df, preds)
    
    assert "composite_score" in optimized.columns
    assert "desirability_index" in optimized.columns
    assert "is_pareto_optimal" in optimized.columns
    assert optimized["is_pareto_optimal"].sum() > 0
    assert (optimized["composite_score"].diff().dropna() <= 0).all()  # sorted descending

def test_chemical_assessments():
    lipinski_pass = assess_lipinski_rule_of_5(mw=400.0, logp=3.0, hbd=2, hba=5)
    assert lipinski_pass["num_violations"] == 0
    assert lipinski_pass["is_lipinski_compliant"] is True
    
    lipinski_fail = assess_lipinski_rule_of_5(mw=550.0, logp=6.2, hbd=6, hba=12)
    assert lipinski_fail["num_violations"] >= 3
    assert lipinski_fail["is_lipinski_compliant"] is False
    
    svg = generate_compound_svg("OPT-001", "Lead-A", "Quinazoline", 446.9, 3.6)
    assert "<svg" in svg
    assert "OPT-001" in svg

def test_ncbi_gene_fetch():
    summary = fetch_gene_summary("1956")
    assert summary["name"] == "EGFR"
    assert "epidermal growth factor receptor" in summary["description"].lower()

if __name__ == "__main__":
    print("Running OptiDrug test suite...")
    test_dataset_generation()
    print("[PASS] test_dataset_generation passed")
    test_preprocessing_pipeline()
    print("[PASS] test_preprocessing_pipeline passed")
    test_model_training_and_metrics()
    print("[PASS] test_model_training_and_metrics passed")
    test_multi_objective_optimization()
    print("[PASS] test_multi_objective_optimization passed")
    test_chemical_assessments()
    print("[PASS] test_chemical_assessments passed")
    test_ncbi_gene_fetch()
    print("[PASS] test_ncbi_gene_fetch passed")
    print("All tests passed successfully!")
