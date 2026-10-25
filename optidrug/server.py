"""
OptiDrug - Modern Localhost Web Platform Server (Flask Backend)
Serves REST API for bioinformatics pipeline, Random Forest QSAR modeling,
Multi-Objective Optimization, NCBI literature queries, and 3D DNA mutation simulation.
"""

import os
import sys
from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd
import numpy as np

# Load local environment
env_file = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_file):
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from data.sample_data import TARGET_CATALOG, generate_target_dataset
from data.ncbi_client import fetch_gene_summary, fetch_recent_pubmed_articles
from pipeline.preprocessing import DataPreprocessor
from pipeline.model import BioactivityPredictor
from pipeline.optimizer import MultiObjectiveOptimizer
from utils.chemical import assess_lipinski_rule_of_5, assess_veber_rules, generate_compound_svg

app = Flask(__name__, template_folder="templates", static_folder="static")

# In-memory cache for fast recalculation
CACHE = {}

@app.route("/")
def index():
    """Serves the interactive single-page bioinformatics dashboard."""
    return render_template("index.html")

@app.route("/api/targets", methods=["GET"])
def get_targets():
    """Returns list of pre-configured biological targets."""
    targets = []
    for k, v in TARGET_CATALOG.items():
        targets.append({
            "target_id": v["target_id"],
            "disease": v["disease"],
            "gene_symbol": v["gene_symbol"],
            "protein_name": v["protein_name"],
            "uniprot_id": v["uniprot_id"],
            "ncbi_gene_id": v["ncbi_gene_id"],
            "dna_simulation": v.get("dna_simulation", {})
        })
    return jsonify({"status": "success", "targets": targets})

@app.route("/api/target/<target_id>", methods=["GET"])
def get_target_details(target_id):
    """Returns detailed target biology, NCBI Gene record, and recent PubMed literature."""
    if target_id not in TARGET_CATALOG:
        target_id = "EGFR"
    info = TARGET_CATALOG[target_id]
    
    # Fetch live NCBI Gene summary & PubMed
    ncbi_summary = fetch_gene_summary(info["ncbi_gene_id"])
    pubmed_articles = fetch_recent_pubmed_articles(info["gene_symbol"], info["disease"], max_results=4)
    
    return jsonify({
        "status": "success",
        "target": info,
        "ncbi_gene": ncbi_summary,
        "pubmed_articles": pubmed_articles
    })

@app.route("/api/pipeline/run", methods=["POST"])
def run_pipeline():
    """
    Executes the full OptiDrug Discovery Pipeline:
    Data Generation -> Preprocessing -> Random Forest Regression -> Multi-Objective Optimization
    """
    data = request.json or {}
    target_id = data.get("target_id", "EGFR")
    if target_id not in TARGET_CATALOG:
        target_id = "EGFR"
        
    n_samples = int(data.get("n_samples", 250))
    seed = int(data.get("seed", 42))
    
    w_eff = float(data.get("w_eff", 0.35))
    w_safe = float(data.get("w_safe", 0.25))
    w_inter = float(data.get("w_inter", 0.15))
    w_cost = float(data.get("w_cost", 0.15))
    w_drug = float(data.get("w_drug", 0.10))
    
    # 1. Dataset Generation
    raw_df = generate_target_dataset(target_id=target_id, n_samples=n_samples, random_state=seed)
    
    # 2. Preprocessing Pipeline
    prep = DataPreprocessor(random_state=seed)
    X_train_norm, X_test_norm, y_train, y_test, cleaned_df = prep.fit_transform(raw_df)
    
    # 3. Model Training & Evaluation
    rf = BioactivityPredictor(n_estimators=100, max_depth=6, random_state=seed)
    model_res = rf.train_and_evaluate(X_train_norm, X_test_norm, y_train, y_test)
    
    # 4. Predict on full cleaned candidate set
    X_all_norm = prep.transform_candidates(cleaned_df)
    predicted_scores = rf.predict(X_all_norm)
    
    # 5. Multi-Objective Optimization
    optimizer = MultiObjectiveOptimizer(
        weight_effectiveness=w_eff,
        weight_safety=w_safe,
        weight_interaction=w_inter,
        weight_cost=w_cost,
        weight_druglikeness=w_drug
    )
    optimized_df = optimizer.optimize_candidates(cleaned_df, predicted_scores)
    
    # Cache objects for fast re-weighting
    CACHE[target_id] = {
        "cleaned_df": cleaned_df,
        "predicted_scores": predicted_scores,
        "rf_model": rf,
        "X_all_norm": X_all_norm,
        "prep": prep
    }
    
    # Package Candidates with Lipinski & Explainability
    candidates_list = []
    top_n_explain = min(20, len(optimized_df))
    
    for idx, row in optimized_df.iterrows():
        lip = assess_lipinski_rule_of_5(row["molecular_weight"], row["logp"], row["hbd"], row["hba"])
        veb = assess_veber_rules(row["rotatable_bonds"], row["tpsa"])
        
        cand = {
            "compound_id": row["compound_id"],
            "compound_name": row["compound_name"],
            "scaffold_type": row["scaffold_type"],
            "molecular_weight": float(row["molecular_weight"]),
            "logp": float(row["logp"]) if pd.notnull(row["logp"]) else None,
            "hbd": int(row["hbd"]),
            "hba": int(row["hba"]),
            "tpsa": float(row["tpsa"]) if pd.notnull(row["tpsa"]) else None,
            "rotatable_bonds": int(row["rotatable_bonds"]),
            "predicted_activity": float(row["predicted_activity"]),
            "predicted_pic50": float(row["predicted_pic50"]),
            "safety_risk": float(row["safety_risk"]),
            "cyp_interaction_risk": float(row["cyp_interaction_risk"]),
            "synthetic_cost_score": float(row["synthetic_cost_score"]),
            "composite_score": float(row["composite_score"]),
            "desirability_index": float(row["desirability_index"]),
            "rule_violations": int(row["rule_violations"]),
            "is_pareto_optimal": bool(row["is_pareto_optimal"]),
            "preliminary_rank": int(row["preliminary_rank"]),
            "optimized_rank": int(row["optimized_rank"]),
            "rank_delta": int(row["rank_delta"]),
            "lipinski_status": lip["status"],
            "lipinski_badge": lip["badge_color"],
            "lipinski_violations": lip["violations"],
            "veber_status": veb["status"],
            "smiles": row["smiles"]
        }
        
        # Local explainability for top candidates
        if idx < top_n_explain:
            norm_series = X_all_norm.loc[row.name]
            cand["explanations"] = rf.explain_candidate(norm_series, top_n=4)
            cand["svg"] = generate_compound_svg(row["compound_id"], row["compound_name"], row["scaffold_type"], row["molecular_weight"], row["logp"] or 3.0)
            
        candidates_list.append(cand)
        
    response_payload = {
        "status": "success",
        "target": TARGET_CATALOG[target_id],
        "audit_log": prep.audit_log,
        "metrics": model_res["metrics"],
        "feature_importances": model_res["feature_importance"].to_dict(orient="records"),
        "evaluation_points": {
            "y_test": [float(v) for v in model_res["y_test"]],
            "y_test_pred": [float(v) for v in model_res["y_test_pred"]]
        },
        "weights": optimizer.weights,
        "candidates": candidates_list
    }
    return jsonify(response_payload)

@app.route("/api/pipeline/reweight", methods=["POST"])
def reweight():
    """Instantly recalculates multi-objective scores & Pareto rankings without retraining model."""
    data = request.json or {}
    target_id = data.get("target_id", "EGFR")
    if target_id not in CACHE:
        return jsonify({"status": "error", "message": "Pipeline not yet run for this target."}), 400
        
    w_eff = float(data.get("w_eff", 0.35))
    w_safe = float(data.get("w_safe", 0.25))
    w_inter = float(data.get("w_inter", 0.15))
    w_cost = float(data.get("w_cost", 0.15))
    w_drug = float(data.get("w_drug", 0.10))
    
    cached = CACHE[target_id]
    cleaned_df = cached["cleaned_df"]
    predicted_scores = cached["predicted_scores"]
    
    optimizer = MultiObjectiveOptimizer(
        weight_effectiveness=w_eff,
        weight_safety=w_safe,
        weight_interaction=w_inter,
        weight_cost=w_cost,
        weight_druglikeness=w_drug
    )
    optimized_df = optimizer.optimize_candidates(cleaned_df, predicted_scores)
    
    results = []
    for idx, row in optimized_df.iterrows():
        results.append({
            "compound_id": row["compound_id"],
            "optimized_rank": int(row["optimized_rank"]),
            "preliminary_rank": int(row["preliminary_rank"]),
            "rank_delta": int(row["rank_delta"]),
            "composite_score": float(row["composite_score"]),
            "desirability_index": float(row["desirability_index"]),
            "is_pareto_optimal": bool(row["is_pareto_optimal"])
        })
    return jsonify({"status": "success", "candidates": results, "weights": optimizer.weights})

if __name__ == "__main__":
    print("Starting OptiDrug Localhost Web Platform on http://localhost:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
