"""
OptiDrug - Data Preprocessing Pipeline
Handles duplicate removal, missing value imputation, categorical encoding,
feature normalization (0-1 scale), and reproducible 80/20 train/test splitting.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    """
    Bioinformatics Preprocessing Pipeline for Molecular QSAR Descriptors.
    Ensures strict separation between train and test distributions to avoid data leakage.
    """
    
    DEFAULT_NUMERIC_FEATURES = [
        "molecular_weight",
        "logp",
        "hbd",
        "hba",
        "tpsa",
        "rotatable_bonds"
    ]
    
    DEFAULT_CATEGORICAL_FEATURES = [
        "scaffold_type"
    ]
    
    TARGET_COLUMN = "activity_score"
    
    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        target_col: str = "activity_score",
        impute_strategy: str = "median",
        test_size: float = 0.20,
        random_state: int = 42
    ):
        self.numeric_features = numeric_features or self.DEFAULT_NUMERIC_FEATURES.copy()
        self.categorical_features = categorical_features or self.DEFAULT_CATEGORICAL_FEATURES.copy()
        self.target_col = target_col
        self.impute_strategy = impute_strategy
        self.test_size = test_size
        self.random_state = random_state
        
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.category_mapping: Dict[str, Dict[str, int]] = {}
        self.processed_feature_names: List[str] = []
        
        # Audit log for dashboard presentation
        self.audit_log: Dict[str, Any] = {
            "initial_rows": 0,
            "duplicate_rows_removed": 0,
            "missing_values_imputed": {},
            "encoded_categories": {},
            "train_rows": 0,
            "test_rows": 0,
            "feature_names": []
        }
        self.is_fitted = False

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identifies and drops duplicate compound rows."""
        initial_len = len(df)
        # Drop duplicates by compound_id if present, else by molecular properties
        if "compound_id" in df.columns:
            dedup_df = df.drop_duplicates(subset=["compound_id"], keep="first").copy()
        else:
            dedup_df = df.drop_duplicates(subset=self.numeric_features, keep="first").copy()
        
        removed = initial_len - len(dedup_df)
        self.audit_log["initial_rows"] = initial_len
        self.audit_log["duplicate_rows_removed"] = removed
        return dedup_df

    def fit_transform(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        """
        Executes full preprocessing:
        1. Duplicate removal
        2. 80/20 train/test split
        3. Imputation (fit on train, applied to both)
        4. Categorical encoding
        5. Min-Max normalization to 0-1 scale
        Returns:
            X_train_norm, X_test_norm, y_train, y_test, cleaned_full_df
        """
        cleaned_df = self.remove_duplicates(df)
        
        # Track missing values before imputation
        missing_counts = cleaned_df[self.numeric_features].isnull().sum().to_dict()
        self.audit_log["missing_values_imputed"] = {k: int(v) for k, v in missing_counts.items() if v > 0}
        
        # Ensure target column exists
        if self.target_col not in cleaned_df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataframe.")
            
        # Split 80/20
        train_df, test_df = train_test_split(
            cleaned_df,
            test_size=self.test_size,
            random_state=self.random_state
        )
        self.audit_log["train_rows"] = len(train_df)
        self.audit_log["test_rows"] = len(test_df)
        
        # 1. Imputation on numeric features
        self.imputer = SimpleImputer(strategy=self.impute_strategy)
        X_train_num = self.imputer.fit_transform(train_df[self.numeric_features])
        X_test_num = self.imputer.transform(test_df[self.numeric_features])
        
        # 2. Categorical Encoding (One-Hot or Frequency/Ordinal)
        # We perform One-Hot Encoding for interpretability
        encoded_train_cats = []
        encoded_test_cats = []
        cat_feature_names = []
        
        for cat_col in self.categorical_features:
            if cat_col in cleaned_df.columns:
                # Find all unique categories from training
                unique_cats = sorted(train_df[cat_col].dropna().unique())
                self.audit_log["encoded_categories"][cat_col] = list(unique_cats)
                
                for cat in unique_cats:
                    col_name = f"{cat_col}_{cat}"
                    cat_feature_names.append(col_name)
                    encoded_train_cats.append((train_df[cat_col] == cat).astype(float).values[:, None])
                    encoded_test_cats.append((test_df[cat_col] == cat).astype(float).values[:, None])
        
        # Combine numeric and categorical features
        if encoded_train_cats:
            X_train_cat_mat = np.hstack(encoded_train_cats)
            X_test_cat_mat = np.hstack(encoded_test_cats)
            X_train_raw = np.hstack([X_train_num, X_train_cat_mat])
            X_test_raw = np.hstack([X_test_num, X_test_cat_mat])
        else:
            X_train_raw = X_train_num
            X_test_raw = X_test_num
            
        all_feature_names = self.numeric_features + cat_feature_names
        self.processed_feature_names = all_feature_names
        self.audit_log["feature_names"] = all_feature_names
        
        # 3. Min-Max Normalization to [0, 1] scale (fit on train, transform test)
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_test_scaled = self.scaler.transform(X_test_raw)
        
        # Convert to DataFrames for clear columns
        X_train_norm = pd.DataFrame(X_train_scaled, columns=all_feature_names, index=train_df.index)
        X_test_norm = pd.DataFrame(X_test_scaled, columns=all_feature_names, index=test_df.index)
        
        y_train = train_df[self.target_col]
        y_test = test_df[self.target_col]
        
        self.is_fitted = True
        return X_train_norm, X_test_norm, y_train, y_test, cleaned_df

    def transform_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms any set of candidate compounds using the fitted imputer and scaler.
        """
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted using fit_transform before transforming candidates.")
            
        X_num = self.imputer.transform(df[self.numeric_features])
        
        encoded_cats = []
        for cat_col in self.categorical_features:
            if cat_col in self.audit_log["encoded_categories"]:
                unique_cats = self.audit_log["encoded_categories"][cat_col]
                for cat in unique_cats:
                    encoded_cats.append((df[cat_col] == cat).astype(float).values[:, None])
                    
        if encoded_cats:
            X_cat_mat = np.hstack(encoded_cats)
            X_raw = np.hstack([X_num, X_cat_mat])
        else:
            X_raw = X_num
            
        X_scaled = self.scaler.transform(X_raw)
        return pd.DataFrame(X_scaled, columns=self.processed_feature_names, index=df.index)
