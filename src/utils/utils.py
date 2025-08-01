
"""utils.py
---------------------------------
Five utility functions that implement the 4‑stage (0–3) fast–to–slow
‘prove‑it’ funnel we discussed, plus a convenience wrapper that runs the
whole sequence.

Stages
-------
0. zero_cost_filter      – residual ↔ feature correlation or MI
1. warm_start_gain       – append N trees to an existing model
2. mini_sample_cv        – 3‑fold CV on ≈5 % query groups
3. full_cv               – production cross‑validation
4. evaluate_feature      – orchestrator that calls 0→3

All heavy artefacts (DMatrix cache pages, baseline booster) are expected
to live on disk so RAM stays constant.

You can import any single helper or call `evaluate_feature` directly.

Example
-------
>>> import xgboost as xgb, pandas as pd, numpy as np
>>> from feature_eval_funnel import evaluate_feature
>>> meta = evaluate_feature(
...     new_feature_series=df['new_col'],
...     full_df=df,
...     group_col='query_id',
...     label_col='label',
...     baseline_model='baseline.json',
...     params={'objective': 'rank:pairwise', 'tree_method': 'hist'},
... )
>>> print(meta['decision'])   # 'pass' or reason for failure
"""

from __future__ import annotations
from src.metric import hitrate_at_3
import json
from typing import Dict, List, Tuple
import polars as pl
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import ndcg_score
import pandas as pd
import os
from sklearn.model_selection import GroupKFold
def get_group_sizes(ranker_ids: np.ndarray) -> np.ndarray:
    # unique in order of first appearance + their counts
    uniq, idx, counts = np.unique(ranker_ids, return_index=True, return_counts=True)
    return counts[np.argsort(idx)]



# ------------------------------------------------------------------
# 0 · Zero‑cost correlation / MI gate
# ------------------------------------------------------------------
def zero_cost_filter(
    residuals: np.ndarray,
    feature: np.ndarray,
    threshold: float = 0.01,
    method: str = "pearson",
) -> Tuple[bool, float]:
    """Check marginal association between residuals and the new column.

    Parameters
    ----------
    residuals : np.ndarray
        y_val – model.predict(x_val)
    feature : np.ndarray
        The candidate feature on the same validation slice.
    threshold : float, optional
        Minimum |corr| (or MI) required to *pass* the gate.
    method : {'pearson', 'spearman'}, optional
        Which statistic to use.

    Returns
    -------
    pass_gate : bool
        True if abs(stat) >= threshold.
    score : float
        The absolute correlation (or MI) value.
    """
    if method == "pearson":
        corr = np.corrcoef(residuals, feature)[0, 1]
    elif method == "spearman":
        corr, _ = spearmanr(residuals, feature)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")
    score = abs(corr)
    return score >= threshold, score


# ------------------------------------------------------------------
# 1 · Warm‑start gain gate
# ------------------------------------------------------------------
def small_retrain(dtrain_full: xgb.DMatrix,
                    dval_base: xgb.DMatrix,
                    dval_new: xgb.DMatrix,
                    group_val,
                    baseline_model_path: str,
                    params: Dict,
                    *,
                    num_boost_round: int = 30,
                    sigma_threshold: float = 0.005,
                    maximize_metric: bool = True) -> Tuple[bool, float]:
    """Append *num_boost_round* trees to the baseline model.

    Parameters
    ----------
    dtrain_full : xgb.DMatrix
        Full training data **including** the new column.
    dval : xgb.DMatrix
        Validation split (or full OOF slice) for metric evaluation.
    baseline_model_path : str
        Path to `baseline.json`.
    params : Dict
        Usual XGBoost params dict. Must include the same objective as baseline.
    num_boost_round : int, default 50
        Number of extra trees to grow.
    sigma_threshold : float
        Minimum delta over baseline metric to pass.
    maximize_metric : bool
        True if higher metric is better (e.g., NDCG), False otherwise.

    Returns
    -------
    pass_gate : bool
        True if improvement ≥ sigma_threshold.
    delta_metric : float
        Measured change in validation metric.
    """
    #booster = xgb.Booster()
    #.load_model(baseline_model_path)
    #preds = booster.predict(dval_base)
    #base_score = hitrate_at_3(dval_base.get_label(), preds, group_val)
    cv_stats = pd.read_csv('model/cv_results.csv')
    base_score = np.mean(cv_stats['val-top@3'])
    base_score_ncdg3 = np.mean(cv_stats['val-ndcg@3'])
    evals_result = {}  
    # --- train a tiny model *from scratch* ------------------------------
    tiny_model = xgb.train(
        params,
        dtrain_full,
        evals=[(dtrain_full, 'train'), (dval_new, 'val')],
        num_boost_round=num_boost_round,
        evals_result=evals_result,
        verbose_eval=False,
    )
    
    #new_score = float(tiny_model.eval_set([(dval_new, "val")], feval=hit3_feval).split(":")[2])
    preds = tiny_model.predict(dval_new)
    new_score = hitrate_at_3(dval_new.get_label(), preds, group_val)
    new_score_ndcg3 = np.mean(evals_result['val']['ndcg@3'])
    delta = new_score - base_score if maximize_metric else base_score - new_score
    print('top@3 before - ', round(base_score, 4), ' top@3 after - ', round(new_score, 4))
    print('ndcg@3 before - ', round(base_score_ncdg3, 4), ' ndcg@3 after - ', round(new_score_ndcg3, 4))
    return delta >= sigma_threshold, delta


# ------------------------------------------------------------------
# 2 · Mini‑sample 3‑fold CV gate
# ------------------------------------------------------------------
def get_group_sizes(ranker_ids: np.ndarray) -> np.ndarray:
    # unique in order of first appearance + their counts
    uniq, idx, counts = np.unique(ranker_ids, return_index=True, return_counts=True)
    return counts[np.argsort(idx)]


def cv(data: pl.DataFrame,
        features: list,
        label_col: str,
        group_col: str,
        params: dict,
        num_boost_round: int,
        ):
    ''' Train new base mode, save used features and metrics in json'''

    feature_cols_all = [c for c in features if c not in {label_col, group_col}]


    X_all = data.select(feature_cols_all).to_numpy()
    y_all = data.select(label_col).to_numpy().ravel()
    grp_all = data.select(group_col).to_numpy().ravel()

    gkf = GroupKFold(n_splits=5)
    cv_records = []
    evals_result = {}  
    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X_all, y_all, groups=grp_all)
    ):
        # Training fold
        dtr = xgb.DMatrix(
            X_all[train_idx],
            label=y_all[train_idx],
            group=get_group_sizes(grp_all[train_idx])
        )
        # Test fold
        X_te = X_all[test_idx]
        y_te = y_all[test_idx]
        grp_te = grp_all[test_idx]

        dte = xgb.DMatrix(
            X_te, label=y_te,
            group=get_group_sizes(grp_te)
        )

        # Train
        bst = xgb.train(
            params,
            dtr,
            evals=[(dtr, 'train'), (dte, 'val')],
            num_boost_round=num_boost_round,
            evals_result=evals_result,
            verbose_eval=False
        )
        preds = bst.predict(dte)
        score = hitrate_at_3(y_te, preds, grp_te)
        
        
        cv_records.append({"fold": fold, "val-top@3": float(score), 'val-ndcg@3': np.mean(evals_result["val"]["ndcg@3"])})

    df = pl.DataFrame(cv_records)
    print('mean top@3 - ', round(df['val-top@3'].mean(), 4))
    print('std top@3 - ', round(df['val-top@3'].std(), 4))
    print('mean ndcg@3 - ', round(df['val-ndcg@3'].mean(), 4))
    print('std ndcg@3 - ', round(df['val-ndcg@3'].std(), 4))
    return df


# ------------------------------------------------------------------
# 3 · Full CV (expensive) -------------------------------------------------
def full_cv(
    dtrain_full: xgb.DMatrix,
    params: Dict,
    folds,
    es_rounds: int = 50,
    metric_name: str = "ndcg@3",
):
    """Run production‑grade CV and return the entire result DataFrame."""
    cv = xgb.cv(
        params,
        dtrain_full,
        num_boost_round=1000,
        folds=folds,
        early_stopping_rounds=es_rounds,
        verbose_eval=50,
        reuse_dmatrix=True,
    )
    return cv


# ------------------------------------------------------------------
# 4 · Orchestrator --------------------------------------------------------
def evaluate_feature(
    full_df: pl.DataFrame,
    new_feature_name: str,
    group_col: str,
    label_col: str,
    baseline_model: str,
    params: Dict,
    corr_threshold: float = 0.01,
    sigma0: float = 0.005,
    sigma1: float = 0.01,
    cat_features_final: list = [],
) -> Dict:
    """Run all gates 0→3 and return a metadata dict.

    The caller must ensure *new_feature_series* is already present in
    *full_df* (or join it beforehand).

    Returns
    -------
    meta : dict
        {
            'decision': 'pass' | 'fail: reason',
            'corr': float,
            'warm_delta': float | None,
            'mini_delta': float | None,
        }
    """
    meta = {"corr": None, "warm_delta": None, "mini_delta": None}

    # ------------------------------------------------------------------
    # | 0 |  residual gate
    # ------------------------------------------------------------------
    # Compute residuals on a hold‑out slice (we use 20 % of groups)



    #val_groups = rng.choice(groups, size=max(1, int(0.2 * len(groups))), replace=False)
    val_groups = pl.read_csv("model/val_ranker_ids.csv")['ranker_id']
    data_xgb = full_df.with_columns([
                                    (pl.col(c).rank("dense") - 1)
                                    .fill_null(-1)
                                    .cast(pl.Int32)
                                    .alias(c)
                                    for c in cat_features_final
                                ])
    val_df = data_xgb.filter(pl.col(group_col).is_in(val_groups))
    train_df = data_xgb.filter(~pl.col(group_col).is_in(val_groups))
    y_val        = val_df.select(label_col).to_numpy().ravel()
    y_train        = train_df.select(label_col).to_numpy().ravel()
    # Baseline booster for residuals
    booster = xgb.Booster()
    booster.load_model(baseline_model)

    feature_cols_all = [c for c in data_xgb.columns if c not in {label_col, group_col, new_feature_name}]
    dval_base = xgb.DMatrix(val_df.select(feature_cols_all).to_numpy(), feature_names=feature_cols_all, group=get_group_sizes(val_df.select(group_col)), label=y_val)
    residuals = val_df.select(label_col).to_numpy().ravel() - booster.predict(dval_base)

    feat_vals = val_df.select(new_feature_name).to_numpy().ravel()
    passed, corr = zero_cost_filter(residuals, feat_vals, threshold=corr_threshold)
    meta["corr"] = corr
    # ------------------------------------------------------------------
    # | 1 |  warm‑start
    # ------------------------------------------------------------------
    # ---------------- warm-start ------------------------------------
    all_feature_cols = [c for c in data_xgb.columns if c not in {label_col, group_col}]

    '''dtrain_full = xgb.DMatrix(train_df.select(all_feature_cols).to_numpy(),
                              label=y_train,
                              group=get_group_sizes(train_df.select(group_col)),
                              feature_names=all_feature_cols)

    dval_new = xgb.DMatrix(val_df.select(all_feature_cols).to_numpy(),
                           label=y_val,
                           group=get_group_sizes(val_df.select(group_col)),
                           feature_names=all_feature_cols)

    passed, warm_delta = small_retrain(dtrain_full,
                                         dval_base,
                                         dval_new,
                                         val_df.select(group_col),
                                         baseline_model,
                                         params,
                                         sigma_threshold=sigma0)
    meta["warm_delta"] = warm_delta
    '''
    # ------------------------------------------------------------------
    # | 2 |  mini CV
    # ------------------------------------------------------------------
    df = cv(
        data_xgb,
        features=all_feature_cols,
        label_col=label_col,
        group_col=group_col,
        params=params,
        num_boost_round=30
    )

    return meta, df
