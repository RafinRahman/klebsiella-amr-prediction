"""
02_model.py
Klebsiella pneumoniae AMR Phenotype Prediction — Model Training and Evaluation
Rahman MR, ideSHi, Dhaka, Bangladesh

Trains LR, RF, XGBoost, LightGBM on carbapenem and 3GC resistance prediction.
Evaluates globally and stratified by South Asia vs Global region.
"""

import pandas as pd
import numpy as np
import json
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              brier_score_loss, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

os.makedirs('outputs', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("Loading preprocessed data...")
X = pd.read_csv('/home/claude/klebsiella_amr/data/kleb_features.csv')
meta = pd.read_csv('/home/claude/klebsiella_amr/data/kleb_metadata.csv')
with open('/home/claude/klebsiella_amr/data/gene_list.json') as f:
    gene_list = json.load(f)

print(f"Feature matrix: {X.shape}")
print(f"Metadata: {meta.shape}")

# Align
assert len(X) == len(meta), "Mismatch between features and metadata"

# ── Subsampling for efficiency (stratified) ──────────────────────────
# Full dataset is 145k rows — use 40k for initial cross-validation
# South Asian isolates are kept fully (5,919), global is sampled
np.random.seed(42)
sa_idx = meta[meta['region'] == 'South Asia'].index.tolist()
global_idx = meta[meta['region'] == 'Global'].index.tolist()

# Sample 34,000 from global to get ~40k total
global_sample = np.random.choice(global_idx, size=34000, replace=False).tolist()
train_idx = sa_idx + global_sample

X_train = X.loc[train_idx].reset_index(drop=True)
meta_train = meta.loc[train_idx].reset_index(drop=True)

print(f"\nWorking dataset: {len(X_train):,} isolates")
print(f"  South Asia: {(meta_train['region']=='South Asia').sum():,}")
print(f"  Global: {(meta_train['region']=='Global').sum():,}")

def evaluate_model(model, X_data, y_data, model_name, target_name, cv_folds=5):
    """Run stratified cross-validation and return metrics dict."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    aucs, auprs, f1s, briers = [], [], [], []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_data, y_data)):
        X_tr, X_val = X_data.iloc[tr_idx], X_data.iloc[val_idx]
        y_tr, y_val = y_data.iloc[tr_idx], y_data.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_val, y_prob))
        auprs.append(average_precision_score(y_val, y_prob))
        f1s.append(f1_score(y_val, y_pred))
        briers.append(brier_score_loss(y_val, y_prob))

    return {
        'model': model_name,
        'target': target_name,
        'cv_auc_mean': np.mean(aucs),
        'cv_auc_std': np.std(aucs),
        'cv_aupr_mean': np.mean(auprs),
        'cv_aupr_std': np.std(auprs),
        'cv_f1_mean': np.mean(f1s),
        'cv_f1_std': np.std(f1s),
        'cv_brier_mean': np.mean(briers),
        'cv_brier_std': np.std(briers),
    }

# ── Models ────────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(
        C=1.0, max_iter=1000, solver='saga',
        class_weight='balanced', random_state=42, n_jobs=-1),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=1, use_label_encoder=False,
        eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        class_weight='balanced', random_state=42, n_jobs=-1,
        verbose=-1),
}

targets = {
    'carbapenem': 'carbapenem_resistant',
    '3gc':        'threegc_resistant',
}

all_results = []

# ── 1. Global cross-validation ────────────────────────────────────────
print("\n=== GLOBAL CROSS-VALIDATION (5-fold) ===")
for target_key, target_col in targets.items():
    y = meta_train[target_col]
    print(f"\nTarget: {target_col} | Prevalence: {y.mean()*100:.1f}%")

    for mname, model in models.items():
        print(f"  [{mname}] training...", end=' ', flush=True)
        res = evaluate_model(model, X_train, y, mname, target_key)
        all_results.append(res)
        print(f"AUC={res['cv_auc_mean']:.3f} ± {res['cv_auc_std']:.3f}  "
              f"AUPR={res['cv_aupr_mean']:.3f}  "
              f"F1={res['cv_f1_mean']:.3f}")

# ── 2. Regional stratified evaluation ─────────────────────────────────
print("\n=== REGIONAL STRATIFIED EVALUATION ===")
print("Train on Global, Test on South Asia (cross-regional generalisation)\n")

# Use all South Asian isolates for testing
X_sa  = X.loc[sa_idx].reset_index(drop=True)
meta_sa = meta.loc[sa_idx].reset_index(drop=True)

# Use all global for training
X_global  = X.loc[global_idx].reset_index(drop=True)
meta_global = meta.loc[global_idx].reset_index(drop=True)

regional_results = []

for target_key, target_col in targets.items():
    y_global = meta_global[target_col]
    y_sa     = meta_sa[target_col]

    print(f"Target: {target_col}")
    print(f"  Global train: {len(y_global):,} | SA test: {len(y_sa):,}")
    print(f"  Global resistance rate: {y_global.mean()*100:.1f}% | SA: {y_sa.mean()*100:.1f}%")

    for mname, model in models.items():
        # Train on full global set
        model.fit(X_global, y_global)

        # Evaluate on South Asia
        y_prob_sa = model.predict_proba(X_sa)[:, 1]
        y_pred_sa = (y_prob_sa >= 0.5).astype(int)

        auc_sa   = roc_auc_score(y_sa, y_prob_sa)
        aupr_sa  = average_precision_score(y_sa, y_prob_sa)
        f1_sa    = f1_score(y_sa, y_pred_sa)
        brier_sa = brier_score_loss(y_sa, y_prob_sa)
        sens_sa  = recall_score(y_sa, y_pred_sa)
        spec_sa  = recall_score(y_sa, y_pred_sa, pos_label=0)

        # Also evaluate on global holdout (20%)
        np.random.seed(99)
        holdout_idx = np.random.choice(len(X_global), size=int(0.2*len(X_global)), replace=False)
        X_gh   = X_global.iloc[holdout_idx]
        y_gh   = y_global.iloc[holdout_idx]
        y_prob_gh = model.predict_proba(X_gh)[:, 1]
        auc_gh = roc_auc_score(y_gh, y_prob_gh)

        regional_results.append({
            'model': mname,
            'target': target_key,
            'auc_global_holdout': auc_gh,
            'auc_south_asia': auc_sa,
            'aupr_south_asia': aupr_sa,
            'f1_south_asia': f1_sa,
            'brier_south_asia': brier_sa,
            'sensitivity_south_asia': sens_sa,
            'specificity_south_asia': spec_sa,
            'auc_delta': auc_gh - auc_sa,
        })

        print(f"  [{mname}] Global-holdout AUC={auc_gh:.3f} | "
              f"South Asia AUC={auc_sa:.3f} | "
              f"Delta={auc_gh - auc_sa:+.3f}")
    print()

# ── 3. Save results ───────────────────────────────────────────────────
cv_df = pd.DataFrame(all_results)
reg_df = pd.DataFrame(regional_results)

cv_df.to_csv('outputs/cv_results.csv', index=False)
reg_df.to_csv('outputs/regional_results.csv', index=False)

print("=== RESULTS SAVED ===")
print(cv_df[['model', 'target', 'cv_auc_mean', 'cv_aupr_mean', 'cv_f1_mean']].to_string(index=False))
print()
print(reg_df[['model', 'target', 'auc_global_holdout', 'auc_south_asia', 'auc_delta']].to_string(index=False))
print("\nDone. Run 03_tune.py next.")
