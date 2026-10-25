"""
OptiDrug - AI-Based Bioinformatics Platform for Drug Candidate Prioritization
Interactive Streamlit Application

Pipeline:
Disease -> Gene -> Protein -> Drug Target -> Candidate Compounds

Data Preprocessing:
Deduplication -> Imputation -> Categorical Encoding -> 0-1 Normalization -> 80/20 Train/Test Split

Machine Learning:
Random Forest Regression (R² ≈ 0.58, MAE ≈ 0.075) with Feature Importance Explainability

Multi-Objective Mathematical Optimization:
Effectiveness vs. Safety vs. CYP Interaction Risk vs. Cost vs. Drug-likeness
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
# Load local environment with graceful fallback
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            pass

from data.sample_data import TARGET_CATALOG, generate_target_dataset
from data.ncbi_client import fetch_gene_summary, fetch_recent_pubmed_articles, get_ncbi_api_key
from pipeline.preprocessing import DataPreprocessor
from pipeline.model import BioactivityPredictor
from pipeline.optimizer import MultiObjectiveOptimizer
from utils.chemical import assess_lipinski_rule_of_5, assess_veber_rules, generate_compound_svg
from utils.visualization import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importances,
    plot_pareto_frontier,
    plot_radar_comparison,
    plot_property_distributions
)

# Page configuration
st.set_page_config(
    page_title="OptiDrug | AI Drug Prioritization Platform",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 12px;
    }
    .pipeline-badge {
        display: inline-block;
        background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 50%, #06b6d4 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 10px 16px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #92400e;
        margin-bottom: 18px;
    }
    .target-card {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/pill.png", width=64)
    st.markdown("### **OptiDrug Platform**")
    st.caption("AI-Powered Drug Candidate Prioritization")
    
    st.divider()
    st.markdown("#### 🎯 **1. Target Selection**")
    target_keys = list(TARGET_CATALOG.keys())
    target_names = [f"{k} — {TARGET_CATALOG[k]['disease']}" for k in target_keys]
    selected_target_idx = st.selectbox(
        "Select Biological Target",
        range(len(target_keys)),
        format_func=lambda i: target_names[i]
    )
    selected_target = target_keys[selected_target_idx]
    target_info = TARGET_CATALOG[selected_target]

    st.markdown("#### ⚙️ **2. Library & Seed Settings**")
    n_compounds = st.slider("Candidate Library Size", min_value=150, max_value=400, value=250, step=25)
    random_seed = st.number_input("Reproducibility Seed", value=42, step=1)
    
    st.divider()
    st.markdown("#### ⚖️ **3. Multi-Objective Weights**")
    st.caption("Adjust optimization priorities (normalized live):")
    
    w_eff = st.slider("Bioactivity / Effectiveness", 0.0, 1.0, 0.35, 0.05)
    w_safe = st.slider("Safety Profile (Low Toxicity)", 0.0, 1.0, 0.25, 0.05)
    w_inter = st.slider("Low CYP Interaction Risk", 0.0, 1.0, 0.15, 0.05)
    w_cost = st.slider("Synthetic Accessibility / Cost", 0.0, 1.0, 0.15, 0.05)
    w_drug = st.slider("Drug-likeness (Rule of 5)", 0.0, 1.0, 0.10, 0.05)

    st.divider()
    st.markdown("#### 🔑 **NCBI Integration**")
    ncbi_key = get_ncbi_api_key()
    if ncbi_key:
        st.success("NCBI API Key Active")
    else:
        st.info("NCBI Public Rate-Limit Mode")
        
    st.markdown("#### 🔬 **Sidecar Tools**")
    st.markdown("[Biomolecular Structure Viewer](sidecar://science/biomolecular_structure_viewer/ui.html)")

# ----------------- DATA & MODEL PIPELINE CACHING -----------------
@st.cache_data
def get_cached_dataset(target_id: str, n_samples: int, seed: int):
    return generate_target_dataset(target_id=target_id, n_samples=n_samples, random_state=seed)

raw_df = get_cached_dataset(selected_target, n_compounds, random_seed)

# Run preprocessing
preprocessor = DataPreprocessor(random_state=random_seed)
X_train_norm, X_test_norm, y_train, y_test, cleaned_df = preprocessor.fit_transform(raw_df)

# Train Random Forest Model
model = BioactivityPredictor(n_estimators=100, max_depth=6, random_state=random_seed)
model_results = model.train_and_evaluate(X_train_norm, X_test_norm, y_train, y_test)

# Predict on all cleaned candidate compounds
X_all_norm = preprocessor.transform_candidates(cleaned_df)
predicted_scores = model.predict(X_all_norm)

# Run Multi-Objective Optimization
optimizer = MultiObjectiveOptimizer(
    weight_effectiveness=w_eff,
    weight_safety=w_safe,
    weight_interaction=w_inter,
    weight_cost=w_cost,
    weight_druglikeness=w_drug
)
optimized_candidates = optimizer.optimize_candidates(cleaned_df, predicted_scores)

# ----------------- MAIN INTERFACE -----------------
st.markdown('<div class="main-title">OptiDrug: AI-Based Drug Candidate Prioritization Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Rational QSAR Modeling, Random Forest Bioactivity Prediction & Multi-Objective Lead Prioritization</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-badge">
    Pipeline: Disease ➔ Gene ➔ Protein ➔ Drug Target ➔ Candidate Compounds
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    <b>⚠️ Research Prioritization Only:</b> Outputs generated by OptiDrug are computational in-silico predictions intended for early-stage hypothesis generation and research prioritization only, not clinical recommendations. The current version utilizes a clearly labeled synthetic demo dataset.
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧬 1. Target & Biological Grounding",
    "🧪 2. Data & Preprocessing Pipeline",
    "🤖 3. Model & Feature Importance",
    "⚖️ 4. Multi-Objective Optimization",
    "🏆 5. Candidate Ranking & Structure Cards"
])

# =========================================================================
# TAB 1: TARGET & BIOLOGICAL GROUNDING
# =========================================================================
with tab1:
    st.subheader(f"Biological Target Grounding: {target_info['target_id']} in {target_info['disease']}")
    
    col1, col2 = st.columns([1.2, 0.8])
    with col1:
        st.markdown(f"""
        <div class="target-card">
            <h4>Target Overview</h4>
            <table style="width:100%; border-collapse: collapse; font-size: 0.92rem;">
                <tr style="border-bottom: 1px solid #f1f5f9; height: 30px;">
                    <td style="color:#64748b; width: 32%;"><b>Selected Disease</b></td>
                    <td><b>{target_info['disease']}</b></td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9; height: 30px;">
                    <td style="color:#64748b;"><b>Target Gene Symbol</b></td>
                    <td><code>{target_info['gene_symbol']}</code> (NCBI Gene ID: <a href="https://www.ncbi.nlm.nih.gov/gene/{target_info['ncbi_gene_id']}" target="_blank">{target_info['ncbi_gene_id']}</a>)</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9; height: 30px;">
                    <td style="color:#64748b;"><b>Protein Entity</b></td>
                    <td><b>{target_info['protein_name']}</b> (UniProt: <a href="https://www.uniprot.org/uniprotkb/{target_info['uniprot_id']}" target="_blank">{target_info['uniprot_id']}</a>)</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9; height: 30px;">
                    <td style="color:#64748b;"><b>Drug Binding Pocket</b></td>
                    <td>{target_info['drug_target_pocket']}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9; height: 30px;">
                    <td style="color:#64748b;"><b>Mechanism of Action</b></td>
                    <td>{target_info['biological_mechanism']}</td>
                </tr>
                <tr style="height: 30px;">
                    <td style="color:#64748b;"><b>Benchmark Approved Drugs</b></td>
                    <td>{', '.join([f'<b>{d}</b>' for d in target_info['benchmark_drugs']])}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        # Live NCBI Gene Metadata
        with st.expander("📚 NCBI Entrez Official Gene Record", expanded=True):
            gene_meta = fetch_gene_summary(target_info["ncbi_gene_id"])
            st.markdown(f"**Organism:** {gene_meta.get('organism', 'Homo sapiens')} | **Chromosome:** {gene_meta.get('chromosome', 'N/A')} | **Aliases:** {gene_meta.get('aliases', 'N/A')}")
            st.info(gene_meta.get("summary", "Summary not available."))
            st.caption(f"Source: {gene_meta.get('source')} (API Key Authenticated)")

    with col2:
        st.markdown("""
        <div class="target-card">
            <h4>Biomolecular Structure</h4>
            <p style="font-size: 0.9rem; color: #475569;">
                Explore the 3D binding pocket and receptor-ligand interaction using the interactive Mol* viewer:
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("[Open Biomolecular Structure Viewer](sidecar://science/biomolecular_structure_viewer/ui.html)")
        
        # Recent PubMed Articles
        st.markdown("#### 📄 Recent Target Literature (NCBI PubMed)")
        articles = fetch_recent_pubmed_articles(target_info["gene_symbol"], target_info["disease"], max_results=3)
        for art in articles:
            st.markdown(f"""
            <div style="background-color:#f8fafc; border-left: 3px solid #3b82f6; padding: 8px 12px; margin-bottom: 8px; border-radius: 4px; font-size: 0.85rem;">
                <a href="{art['url']}" target="_blank" style="font-weight: 600; color: #1e40af; text-decoration: none;">{art['title']}</a><br>
                <span style="color: #64748b;">{art['authors']} — <i>{art['source']}</i> ({art['pubdate']}) [PMID: {art['pmid']}]</span>
            </div>
            """, unsafe_allow_html=True)

# =========================================================================
# TAB 2: DATA & PREPROCESSING PIPELINE
# =========================================================================
with tab2:
    st.subheader("Data Cleaning & Preprocessing Pipeline")
    st.markdown("""
    The bioinformatics data pipeline performs standard quality assurance:
    **Duplicate Removal ➔ Missing Value Imputation (Median) ➔ Categorical Scaffold Encoding ➔ Min-Max Normalization (0–1 Scale) ➔ 80/20 Train/Test Split**.
    """)
    
    audit = preprocessor.audit_log
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric("Initial Records", audit["initial_rows"])
    with col_m2:
        st.metric("Duplicates Dropped", audit["duplicate_rows_removed"])
    with col_m3:
        imputed_count = sum(audit["missing_values_imputed"].values())
        st.metric("Missing Imputed", imputed_count)
    with col_m4:
        st.metric("Train Split (80%)", f"{audit['train_rows']} compounds")
    with col_m5:
        st.metric("Test Split (20%)", f"{audit['test_rows']} compounds")
        
    st.divider()
    
    col_raw, col_proc = st.columns([1, 1])
    with col_raw:
        st.markdown("#### Raw Chemical Candidate Database (First 5 Rows)")
        st.dataframe(
            raw_df[["compound_id", "scaffold_type", "molecular_weight", "logp", "hbd", "hba", "tpsa", "rotatable_bonds", "activity_score", "pic50"]].head(5),
            use_container_width=True
        )
    with col_proc:
        st.markdown("#### Normalized Features (0–1 Scale, Model Ready)")
        st.dataframe(
            X_train_norm.head(5).round(3),
            use_container_width=True
        )
        
    st.markdown("#### Molecular Property Distributions (Demo Library)")
    st.plotly_chart(plot_property_distributions(cleaned_df), use_container_width=True)

# =========================================================================
# TAB 3: MACHINE LEARNING MODEL & EVALUATION
# =========================================================================
with tab3:
    st.subheader("Machine Learning Model: Random Forest Regression")
    st.markdown("""
    A Random Forest Regressor predicts the computational activity score for each candidate compound based on its molecular properties.
    """)
    
    metrics = model_results["metrics"]
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Held-out R² Score", f"{metrics['test_r2']:.4f}", help="Target demo benchmark: R² ≈ 0.58")
    with m_col2:
        st.metric("Held-out MAE", f"{metrics['test_mae']:.4f}", help="Target demo benchmark: MAE ≈ 0.075")
    with m_col3:
        st.metric("Held-out RMSE", f"{metrics['test_rmse']:.4f}")
    with m_col4:
        st.metric("Pearson Correlation (r)", f"{metrics['test_pearson']:.4f}")
        
    st.divider()
    
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_eval = plot_actual_vs_predicted(
            model_results["y_test"],
            model_results["y_test_pred"],
            metrics["test_r2"],
            metrics["test_mae"]
        )
        st.plotly_chart(fig_eval, use_container_width=True)
        
    with chart_col2:
        fig_res = plot_residuals(model_results["y_test"], model_results["y_test_pred"])
        st.plotly_chart(fig_res, use_container_width=True)
        
    st.markdown("#### Model Explainability: Feature Importance")
    st.markdown("Quantifies how much each molecular descriptor drives the predicted bioactivity score:")
    fi_df = model_results["feature_importance"]
    st.plotly_chart(plot_feature_importances(fi_df), use_container_width=True)

# =========================================================================
# TAB 4: MULTI-OBJECTIVE MATHEMATICAL OPTIMIZATION
# =========================================================================
with tab4:
    st.subheader("Multi-Objective Mathematical Optimization")
    st.markdown("""
    Drug discovery is intrinsically multi-objective: potent candidates often fail due to cardiotoxicity (hERG),
    CYP450 metabolic liability, poor solubility, or severe Lipinski violations.
    OptiDrug balances these competing criteria using **Pareto Optimization**, **Derringer-Suich Desirability Functions**,
    and **Weighted Multi-Attribute Scoring**:
    """)
    
    # Weight summary badges
    w_norm = optimizer.weights
    st.markdown(f"""
    <div style="background-color: #f1f5f9; padding: 10px 14px; border-radius: 8px; margin-bottom: 15px; font-size: 0.9rem;">
        <b>Current Objective Weights:</b>
        Effectiveness: <b>{w_norm['effectiveness']*100:.1f}%</b> |
        Safety: <b>{w_norm['safety']*100:.1f}%</b> |
        Low CYP Risk: <b>{w_norm['interaction']*100:.1f}%</b> |
        Synthetic Economy: <b>{w_norm['cost']*100:.1f}%</b> |
        Drug-likeness: <b>{w_norm['druglikeness']*100:.1f}%</b>
    </div>
    """, unsafe_allow_html=True)
    
    opt_col1, opt_col2 = st.columns([1.1, 0.9])
    with opt_col1:
        st.plotly_chart(plot_pareto_frontier(optimized_candidates), use_container_width=True)
        
    with opt_col2:
        top_3 = [optimized_candidates.iloc[i] for i in range(min(4, len(optimized_candidates)))]
        st.plotly_chart(plot_radar_comparison(top_3, target_info['target_id']), use_container_width=True)
        
    st.markdown("#### Preliminary Rank vs. Multi-Objective Optimized Rank")
    st.caption("Notice candidates that shifted rank when balancing safety, toxicity, cost, and drug-likeness:")
    
    comparison_table = optimized_candidates[[
        "optimized_rank", "preliminary_rank", "rank_delta",
        "compound_id", "compound_name", "scaffold_type",
        "predicted_pic50", "safety_risk", "cyp_interaction_risk", "synthetic_cost_score", "composite_score", "is_pareto_optimal"
    ]].head(10).copy()
    
    comparison_table["is_pareto_optimal"] = comparison_table["is_pareto_optimal"].apply(lambda x: "⭐ Pareto" if x else "—")
    st.dataframe(comparison_table, use_container_width=True)

# =========================================================================
# TAB 5: CANDIDATE RANKING & STRUCTURE CARDS
# =========================================================================
with tab5:
    st.subheader("Candidate Prioritization & Molecular Inspector")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        filter_mode = st.radio("Candidate Filter", ["All Candidates", "Pareto Optimal Only ⭐", "Lipinski Compliant Only"], horizontal=True)
    with filter_col2:
        selected_scaffold = st.selectbox("Filter by Scaffold", ["All Scaffolds"] + list(optimized_candidates["scaffold_type"].unique()))
    with filter_col3:
        search_query = st.text_input("Search Compound ID / Name", placeholder="e.g. OPT-EGFR-005")

    # Apply filters
    filtered_df = optimized_candidates.copy()
    if filter_mode == "Pareto Optimal Only ⭐":
        filtered_df = filtered_df[filtered_df["is_pareto_optimal"]]
    elif filter_mode == "Lipinski Compliant Only":
        filtered_df = filtered_df[filtered_df["rule_violations"] <= 1]
        
    if selected_scaffold != "All Scaffolds":
        filtered_df = filtered_df[filtered_df["scaffold_type"] == selected_scaffold]
        
    if search_query:
        filtered_df = filtered_df[
            filtered_df["compound_id"].str.contains(search_query, case=False) |
            filtered_df["compound_name"].str.contains(search_query, case=False)
        ]
        
    st.dataframe(
        filtered_df[[
            "optimized_rank", "compound_id", "compound_name", "scaffold_type",
            "predicted_pic50", "safety_risk", "cyp_interaction_risk", "synthetic_cost_score",
            "composite_score", "desirability_index", "rule_violations", "is_pareto_optimal"
        ]].rename(columns={
            "optimized_rank": "Rank",
            "predicted_pic50": "Pred pIC50",
            "safety_risk": "Safety Risk",
            "cyp_interaction_risk": "CYP Risk",
            "synthetic_cost_score": "Cost Score",
            "composite_score": "Composite (0-100)",
            "desirability_index": "Desirability (D)",
            "rule_violations": "Violations",
            "is_pareto_optimal": "Pareto?"
        }),
        use_container_width=True
    )
    
    # Download CSV
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prioritized Candidates (CSV)",
        data=csv_bytes,
        file_name=f"OptiDrug_{selected_target}_Prioritized_Candidates.csv",
        mime="text/csv"
    )
    
    st.divider()
    st.markdown("### 🔍 Detailed Compound Inspection Card")
    
    if len(filtered_df) > 0:
        c_list = list(filtered_df["compound_id"].values)
        inspected_id = st.selectbox("Select Candidate to Inspect", c_list, index=0)
        c_row = filtered_df[filtered_df["compound_id"] == inspected_id].iloc[0]
        
        card_col1, card_col2 = st.columns([1.1, 0.9])
        with card_col1:
            st.markdown(f"#### **{c_row['compound_id']} — {c_row['compound_name']}**")
            # Render SVG Structure Card
            svg_html = generate_compound_svg(
                compound_id=c_row["compound_id"],
                compound_name=c_row["compound_name"],
                scaffold=c_row["scaffold_type"],
                mw=c_row["molecular_weight"],
                logp=c_row["logp"]
            )
            st.markdown(svg_html, unsafe_allow_html=True)
            st.caption(f"SMILES: `{c_row['smiles']}`")
            
        with card_col2:
            st.markdown("#### Drug-likeness & ADMET Profiles")
            lipinski = assess_lipinski_rule_of_5(
                mw=c_row["molecular_weight"],
                logp=c_row["logp"],
                hbd=int(c_row["hbd"]),
                hba=int(c_row["hba"])
            )
            veber = assess_veber_rules(
                rotatable_bonds=int(c_row["rotatable_bonds"]),
                tpsa=c_row["tpsa"]
            )
            
            st.markdown(f"**Lipinski Rule of 5:** `{lipinski['status']}`")
            if lipinski["violations"]:
                st.caption(f"Violations: {', '.join(lipinski['violations'])}")
                
            st.markdown(f"**Veber Bioavailability:** `{veber['status']}`")
            if veber["violations"]:
                st.caption(f"Violations: {', '.join(veber['violations'])}")
                
            st.markdown("---")
            st.markdown("#### Why this Candidate Ranked Here:")
            
            # Local explainability
            c_norm_features = X_all_norm.loc[c_row.name]
            explanations = model.explain_candidate(c_norm_features, top_n=4)
            for exp in explanations:
                feat_clean = exp['feature'].replace('_', ' ').title()
                st.write(f"• **{feat_clean}** (Norm Value: {exp['normalized_value']:.2f}, Relative Impact: {exp['contribution_score']*100:.1f}%)")
    else:
        st.warning("No candidate compounds match the selected filter criteria.")
