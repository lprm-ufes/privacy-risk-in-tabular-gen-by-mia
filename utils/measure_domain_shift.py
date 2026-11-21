from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def evaluate_domain_shift(source_df: pd.DataFrame, target_df: pd.DataFrame, num_columns: List[str], cat_columns: List[str], seed: int = 42):
    # Sample equal sizes to keep a balanced 50/50 binary classification
    n_samples = min(len(source_df), len(target_df))
    df_source = source_df[num_columns + cat_columns].sample(n=n_samples, random_state=seed).copy()
    df_target = target_df[num_columns + cat_columns].sample(n=n_samples, random_state=seed).copy()

    df_source["is_target_domain"] = 0
    df_target["is_target_domain"] = 1

    combined = pd.concat([df_source, df_target], ignore_index=True)
    X = combined[num_columns + cat_columns]
    y = combined["is_target_domain"]

    # 2. Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_columns),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # 3. Train Random Forest Domain Classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
    clf.fit(X_train_proc, y_train)

    # 4. Predict probabilities and calculate metrics
    y_prob = clf.predict_proba(X_test_proc)[:, 1]
    y_pred = clf.predict(X_test_proc)

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    error = 1.0 - acc
    # Proxy A-Distance: d_A = 2 * (1 - 2 * error)
    pad = 2 * (1 - 2 * error)

    # 5. Extract top distinguishing features
    feature_names = preprocessor.get_feature_names_out()
    importances = clf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:3]
    top_features = [f"{feature_names[i]} ({importances[i]:.3f})" for i in top_indices]

    print(f"\n{'='*55}")
    print(f"Domain Shift: Source DF vs. Target DF")
    print(f"{'='*55}")
    print(f"  Samples per domain : {n_samples}")
    print(f"  Domain ROC-AUC     : {auc:.4f}")
    print(f"  Accuracy           : {acc:.4f} (Error: {error:.4f})")
    print(f"  Proxy A-Distance   : {pad:.4f}")
    print(f"  Top Discrepancies  : {', '.join(top_features)}")
    return auc, pad