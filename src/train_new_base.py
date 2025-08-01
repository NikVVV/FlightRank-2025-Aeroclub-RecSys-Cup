import polars as pl
import numpy as np
import xgboost as xgb
import pandas as pd
import os
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
from src.metric import hitrate_at_3
def get_group_sizes(ranker_ids: np.ndarray) -> np.ndarray:
    # unique in order of first appearance + their counts
    uniq, idx, counts = np.unique(ranker_ids, return_index=True, return_counts=True)
    return counts[np.argsort(idx)]


def train_base_model(data: pl.DataFrame,
                     features: list,
                     label_col: str,
                     group_col: str,
                     params: dict,
                     num_boost_round: int,
                     baseline_model_path: str,
                     seed: int, 
                     verbose_eval_size: int,
                     full_cv: False,
                     cat_features_final: list = []):
    ''' Train new base mode, save used features and metrics in json'''
    data_xgb = data.with_columns([
                                (pl.col(c).rank("dense") - 1)
                                .fill_null(-1)
                                .cast(pl.Int32)
                                .alias(c)
                                for c in cat_features_final
                            ])
    rng = np.random.default_rng(seed=seed)
    groups = data_xgb.select('ranker_id').unique().to_numpy().ravel()
    val_groups = rng.choice(groups, size=max(1, int(0.01 * len(groups))), replace=False)
    val_df = data_xgb.filter(pl.col('ranker_id').is_in(val_groups))
    train_df = data_xgb.filter(~pl.col('ranker_id').is_in(val_groups))

    train_df['ranker_id'].unique().to_frame().write_csv(baseline_model_path + 'train_ranker_ids.csv')
    val_df['ranker_id'].unique().to_frame().write_csv(baseline_model_path + 'val_ranker_ids.csv')

    y_val        = val_df.select(label_col).to_numpy().ravel()
    y_train        = train_df.select(label_col).to_numpy().ravel()

    feature_cols_all = [c for c in features if c not in {label_col, group_col}]

    dtrain = xgb.DMatrix(train_df.select(feature_cols_all).to_numpy(),
                            label=y_train,
                            group=get_group_sizes(train_df.select(group_col)),
                            feature_names=feature_cols_all)
    
    dval = xgb.DMatrix(val_df.select(feature_cols_all).to_numpy(),
                           label=y_val,
                           group=get_group_sizes(val_df.select(group_col)),
                           feature_names=feature_cols_all)
    
    tiny_model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=verbose_eval_size,
    )
    xgb_importance = tiny_model.get_score(importance_type='gain')
    xgb_importance_df = pl.DataFrame(
        [{'feature': k, 'importance': v} for k, v in xgb_importance.items()]
    ).sort('importance', descending=bool(1))
    xgb_importance_df.write_csv(baseline_model_path + 'importance.csv')
    #print(tiny_model.eval_set([(dval, "val")]))

    preds = tiny_model.predict(dval)
    score = hitrate_at_3(dval.get_label(), preds, val_df.select(group_col).to_series().to_numpy())
    tiny_model.save_model(baseline_model_path + 'base.json')
    print(score)


    #tiny_model.save_config(baseline_model_path + 'config.sjon')
    if full_cv:
        X_all = data_xgb.select(feature_cols_all).to_numpy()
        y_all = data_xgb.select(label_col).to_numpy().ravel()
        grp_all = data_xgb.select(group_col).to_numpy().ravel()

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

        # Save CV results
        
        pl.DataFrame(cv_records).write_csv(
            os.path.join(baseline_model_path, "cv_results.csv")
        )
        df = pl.DataFrame(cv_records)
        print('mean top@3 - ', round(df['val-top@3'].mean(), 4))
        print('std top@3 - ', round(df['val-top@3'].std(), 4))
        print('mean ndcg@3 - ', round(df['val-ndcg@3'].mean(), 4))
        print('std ndcg@3 - ', round(df['val-ndcg@3'].std(), 4))