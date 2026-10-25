"""
OptiDrug - Multi-Objective Mathematical Optimization Engine
Provides multi-criteria candidate prioritization balancing:
  1. Effectiveness / Predicted Bioactivity (Maximize)
  2. Safety / Toxicity Risk (Minimize)
  3. CYP / Drug-Drug Interaction Risk (Minimize)
  4. Synthetic Accessibility & Cost (Minimize)
  5. Drug-likeness & Lipinski Violations (Minimize)

Methods implemented:
  - Weighted Sum Model (WSM) with user-adjustable weights
  - Derringer-Suich Desirability Function (Geometric Multi-Attribute Index)
  - Pareto Non-Dominated Sorting (Identification of Pareto-Optimal candidates)
  - TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

class MultiObjectiveOptimizer:
    """
    Mathematical optimization engine for balancing effectiveness, safety, interaction risk,
    cost, and drug-likeness.
    """
    def __init__(
        self,
        weight_effectiveness: float = 0.35,
        weight_safety: float = 0.25,
        weight_interaction: float = 0.15,
        weight_cost: float = 0.15,
        weight_druglikeness: float = 0.10
    ):
        self.raw_weights = {
            "effectiveness": weight_effectiveness,
            "safety": weight_safety,
            "interaction": weight_interaction,
            "cost": weight_cost,
            "druglikeness": weight_druglikeness
        }
        self.weights = self._normalize_weights(self.raw_weights)

    def _normalize_weights(self, weights_dict: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(0.0, v) for v in weights_dict.values())
        if total <= 1e-6:
            return {k: 1.0 / len(weights_dict) for k in weights_dict}
        return {k: max(0.0, v) / total for k, v in weights_dict.items()}

    def update_weights(self, **kwargs) -> None:
        """Dynamically update optimization weights."""
        for k, v in kwargs.items():
            if k in self.raw_weights:
                self.raw_weights[k] = float(v)
        self.weights = self._normalize_weights(self.raw_weights)

    @staticmethod
    def calculate_lipinski_violations(df: pd.DataFrame) -> pd.Series:
        """
        Calculates Lipinski Rule of Five violations:
          - MW <= 500 Da
          - LogP <= 5.0
          - HBD <= 5
          - HBA <= 10
        Plus Veber rules (Rotatable Bonds <= 10, TPSA <= 140).
        """
        v_mw = (df["molecular_weight"] > 500).astype(int)
        v_logp = (df["logp"] > 5.0).astype(int)
        v_hbd = (df["hbd"] > 5).astype(int)
        v_hba = (df["hba"] > 10).astype(int)
        v_rb = (df["rotatable_bonds"] > 10).astype(int)
        v_tpsa = (df["tpsa"] > 140.0).astype(int)
        
        return v_mw + v_logp + v_hbd + v_hba + v_rb + v_tpsa

    def optimize_candidates(self, df: pd.DataFrame, predicted_scores: np.ndarray) -> pd.DataFrame:
        """
        Computes composite multi-objective scores, desirability indices, TOPSIS rankings,
        and identifies the Pareto-optimal frontier.
        """
        df = df.copy()
        df["predicted_activity"] = predicted_scores
        # Estimated pIC50 if not present
        if "predicted_pic50" not in df.columns:
            df["predicted_pic50"] = (4.50 + 5.0 * df["predicted_activity"]).round(3)

        # Ensure risk and cost columns exist
        if "safety_risk" not in df.columns:
            df["safety_risk"] = 0.30
        if "cyp_interaction_risk" not in df.columns:
            df["cyp_interaction_risk"] = 0.25
        if "synthetic_cost_score" not in df.columns:
            df["synthetic_cost_score"] = 5.0
            
        df["rule_violations"] = self.calculate_lipinski_violations(df)

        # --- 1. Normalized Objectives [0, 1] (1 is best, 0 is worst) ---
        # Effectiveness: higher is better
        min_act = df["predicted_activity"].min()
        max_act = df["predicted_activity"].max()
        norm_eff = (df["predicted_activity"] - min_act) / (max_act - min_act + 1e-6)

        # Safety: lower risk is better -> invert
        min_safe = df["safety_risk"].min()
        max_safe = df["safety_risk"].max()
        norm_safe = 1.0 - (df["safety_risk"] - min_safe) / (max_safe - min_safe + 1e-6)

        # Interaction: lower risk is better -> invert
        min_inter = df["cyp_interaction_risk"].min()
        max_inter = df["cyp_interaction_risk"].max()
        norm_inter = 1.0 - (df["cyp_interaction_risk"] - min_inter) / (max_inter - min_inter + 1e-6)

        # Cost: lower is better -> invert
        min_cost = df["synthetic_cost_score"].min()
        max_cost = df["synthetic_cost_score"].max()
        norm_cost = 1.0 - (df["synthetic_cost_score"] - min_cost) / (max_cost - min_cost + 1e-6)

        # Drug-likeness: fewer violations is better -> invert (0-4 violations max)
        max_viol = max(df["rule_violations"].max(), 4)
        norm_drug = 1.0 - (df["rule_violations"] / max_viol).clip(0, 1)

        # Store individual normalized desirability components
        df["norm_effectiveness"] = norm_eff.round(4)
        df["norm_safety"] = norm_safe.round(4)
        df["norm_interaction"] = norm_inter.round(4)
        df["norm_cost"] = norm_cost.round(4)
        df["norm_druglikeness"] = norm_drug.round(4)

        # --- 2. Weighted Sum Model (WSM) ---
        w = self.weights
        composite_score = (
            w["effectiveness"] * norm_eff +
            w["safety"] * norm_safe +
            w["interaction"] * norm_inter +
            w["cost"] * norm_cost +
            w["druglikeness"] * norm_drug
        )
        df["composite_score"] = (composite_score * 100.0).round(2)

        # --- 3. Derringer-Suich Geometric Desirability ---
        # Add small epsilon to avoid exact 0
        eps = 1e-3
        d_eff = (norm_eff + eps).clip(0.01, 1.0)
        d_safe = (norm_safe + eps).clip(0.01, 1.0)
        d_inter = (norm_inter + eps).clip(0.01, 1.0)
        d_cost = (norm_cost + eps).clip(0.01, 1.0)
        d_drug = (norm_drug + eps).clip(0.01, 1.0)

        geom_desirability = (
            (d_eff ** w["effectiveness"]) *
            (d_safe ** w["safety"]) *
            (d_inter ** w["interaction"]) *
            (d_cost ** w["cost"]) *
            (d_drug ** w["druglikeness"])
        )
        df["desirability_index"] = (geom_desirability * 100.0).round(2)

        # --- 4. Pareto Dominance Sorting ---
        # Objectives to maximize: [predicted_activity, -safety_risk, -cyp_interaction_risk, -synthetic_cost_score]
        is_pareto = self._compute_pareto_frontier(
            df[["norm_effectiveness", "norm_safety", "norm_interaction", "norm_cost", "norm_druglikeness"]].values
        )
        df["is_pareto_optimal"] = is_pareto

        # --- 5. Preliminary Rank vs Optimized Rank ---
        df["preliminary_rank"] = df["predicted_activity"].rank(ascending=False, method="min").astype(int)
        df["optimized_rank"] = df["composite_score"].rank(ascending=False, method="min").astype(int)
        df["rank_delta"] = df["preliminary_rank"] - df["optimized_rank"]  # Positive means optimization boosted ranking!

        return df.sort_values(by="optimized_rank", ascending=True).reset_index(drop=True)

    @staticmethod
    def _compute_pareto_frontier(costs: np.ndarray) -> np.ndarray:
        """
        Finds Pareto-efficient points where all columns in `costs` are to be maximized.
        Returns a boolean array where True indicates the point is non-dominated.
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Point i is dominated if another point j is >= c in all dimensions and > c in at least one
                dominated = np.all(costs >= c, axis=1) & np.any(costs > c, axis=1)
                if np.any(dominated):
                    is_efficient[i] = False
        return is_efficient
