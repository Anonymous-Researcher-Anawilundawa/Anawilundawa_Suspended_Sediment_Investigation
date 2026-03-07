"""
RANDOM FOREST & GRADIENT BOOSTING
================================================================
Leave-One-Plot-Out (LOPO) cross-validation of Random Forest and
Gradient Boosting regressors for SSC prediction.

FEATURE SET
-----------
    |U|          streamwise velocity magnitude
    h            water depth
    |U|·h        discharge proxy
    Δx           spatial step (canal geometry)
    cum_dist     cumulative distance from canal inlet
    HT           tidal state binary flag
    CPC          canal type binary flag
    Location     label-encoded location code

NOTE: Input_Near_Bed_Conc_sb_g_L EXCLUDED — r > 0.999 with target
      (data leakage).

USAGE
-----
    pip install scikit-learn --break-system-packages
    python model_ml_ensemble.py

OUTPUTS
-------
    Model_Output_RandomForest.csv       — LOPO predictions
    Model_Output_GradientBoosting.csv   — LOPO predictions
    Model_Stats_RandomForest.csv        — performance metrics
    Model_Stats_GradientBoosting.csv    — performance metrics
    Model_FeatureImportance.csv         — full-dataset feature importances
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import stats as sp

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    sys.exit("scikit-learn not found. Install with:\n"
             "  pip install scikit-learn --break-system-packages")

# =============================================================================
# PATHS
# =============================================================================
DIR      = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(DIR, 'Input_Data.csv')

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
N_TREES       = 500
RF_MAX_DEPTH  = 4
RF_MIN_LEAF   = 3
GB_MAX_DEPTH  = 3
GB_LR         = 0.05
RANDOM_STATE  = 42
THRESHOLD     = 0.08


# =============================================================================
# HELPERS
# =============================================================================
def classification_stats(m, p, t=THRESHOLD):
    tp=int(((m>=t)&(p>=t)).sum()); fn=int(((m>=t)&(p<t)).sum())
    tn=int(((m<t)&(p<t)).sum());   fp=int(((m<t)&(p>=t)).sum())
    sens = tp/(tp+fn) if tp+fn>0 else 0.0
    spec = tn/(tn+fp) if tn+fp>0 else 0.0
    f1   = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn>0 else 0.0
    acc  = (tp+tn)/len(m)
    return dict(TP=tp, FN=fn, TN=tn, FP=fp,
                Sensitivity=sens, Specificity=spec, F1=f1, Accuracy=acc)


def save_stats(m, p, label, plots, out_path):
    res  = m - p
    n    = len(m)
    rmse = np.sqrt((res**2).mean())
    mae  = np.abs(res).mean()
    bias = res.mean()
    r_p, pv_p = sp.pearsonr(m, p)
    r_s, pv_s = sp.spearmanr(m, p)
    cl = classification_stats(m, p)

    # Per-fold stats
    fold_rows = []
    for plot in sorted(np.unique(plots)):
        mask = plots == plot
        if mask.sum() == 0:
            continue
        r_f  = m[mask] - p[mask]
        cl_f = classification_stats(m[mask], p[mask])
        r_p_f, _ = sp.pearsonr(m[mask], p[mask]) if mask.sum()>2 else (np.nan, None)
        fold_rows.append({
            'Plot': plot, 'n': int(mask.sum()),
            'RMSE': np.sqrt((r_f**2).mean()),
            'Bias': r_f.mean(),
            'Pearson_r': r_p_f,
            **{k: v for k, v in cl_f.items()}
        })

    stats_rows = [
        ('Model',           label),
        ('n_obs',           n),
        ('RMSE_gL',         rmse),
        ('MAE_gL',          mae),
        ('Bias_gL',         bias),
        ('Pearson_r',       r_p),
        ('Pearson_r_pvalue',pv_p),
        ('Spearman_rho',    r_s),
        ('Spearman_rho_pvalue', pv_s),
        ('Threshold_gL',    THRESHOLD),
        ('TP',              cl['TP']),
        ('FN',              cl['FN']),
        ('TN',              cl['TN']),
        ('FP',              cl['FP']),
        ('Sensitivity',     cl['Sensitivity']),
        ('Specificity',     cl['Specificity']),
        ('F1',              cl['F1']),
        ('Accuracy',        cl['Accuracy']),
    ]
    pd.DataFrame(stats_rows, columns=['Metric','Value']).to_csv(out_path, index=False)

    # Per-fold appended as second sheet (separate CSV)
    fold_path = out_path.replace('.csv', '_PerFold.csv')
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    return rmse, r_p, cl


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not os.path.isfile(INPUT_FILE):
        sys.exit(f"ERROR: Cannot find {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    df = df[df['Distance_to_Neighbor_m'] > 0].copy().reset_index(drop=True)
    n  = len(df)
    print(f"Loaded {n} non-seed observations.")

    meas  = df['Target_Measured_Avg_s0_g_L'].values
    plots = df['Plot'].values

    # Cumulative canal distance
    df['cum_dist_m'] = 0.0
    for keys, grp in df.groupby(['Plot', 'Sampling_Day', 'Tide_State']):
        df.loc[grp.index, 'cum_dist_m'] = grp['Distance_to_Neighbor_m'].cumsum().values

    loc_enc     = LabelEncoder().fit(df['Location'].values)
    loc_encoded = loc_enc.transform(df['Location'].values).astype(float)

    U   = np.abs(df['Streamwise_Velocity_U_m_s'].values)
    h   = df['Water_Depth_m'].values
    dx  = df['Distance_to_Neighbor_m'].values
    HT  = (df['Tide_State'] == 'HT').astype(float).values
    CPC = (df['Canal_Type'] == 'CPC').astype(float).values

    X = np.column_stack([
        U, h, U*h, dx,
        df['cum_dist_m'].values,
        HT, CPC, loc_encoded,
    ])
    feat_names = ['|U|', 'h', '|U|·h', 'Δx', 'cum_dist', 'HT', 'CPC', 'Location']

    # ── LOPO cross-validation ────────────────────────────────────────────
    unique_plots = np.unique(plots)

    rf_preds = np.full(n, np.nan)
    gb_preds = np.full(n, np.nan)

    print("\nRunning LOPO cross-validation...")
    for plot in unique_plots:
        train = plots != plot
        test  = plots == plot
        print(f"  Withheld plot {plot}: train n={train.sum()}  test n={test.sum()}")

        rf = RandomForestRegressor(
            n_estimators=N_TREES, max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_LEAF, random_state=RANDOM_STATE, n_jobs=-1
        )
        gb = GradientBoostingRegressor(
            n_estimators=N_TREES, max_depth=GB_MAX_DEPTH,
            learning_rate=GB_LR, random_state=RANDOM_STATE
        )
        rf.fit(X[train], meas[train])
        gb.fit(X[train], meas[train])
        rf_preds[test] = rf.predict(X[test])
        gb_preds[test] = gb.predict(X[test])

    # ── Full-dataset models for feature importance ───────────────────────
    print("\nFitting full-dataset models for feature importance...")
    rf_full = RandomForestRegressor(
        n_estimators=N_TREES, max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_LEAF, random_state=RANDOM_STATE, n_jobs=-1
    )
    gb_full = GradientBoostingRegressor(
        n_estimators=N_TREES, max_depth=GB_MAX_DEPTH,
        learning_rate=GB_LR, random_state=RANDOM_STATE
    )
    rf_full.fit(X, meas); gb_full.fit(X, meas)

    fi = pd.DataFrame({
        'Feature': feat_names,
        'RF_Importance':  rf_full.feature_importances_,
        'GB_Importance':  gb_full.feature_importances_,
    }).sort_values('RF_Importance', ascending=False)
    fi_path = os.path.join(DIR, 'Model_FeatureImportance.csv')
    fi.to_csv(fi_path, index=False)
    print(f"Feature importances → {fi_path}")

    # ── Save prediction CSVs ─────────────────────────────────────────────
    out_rf = df.copy()
    out_rf['Modelled_s0_RF']  = rf_preds
    out_rf['Residual']        = meas - rf_preds
    out_rf.to_csv(os.path.join(DIR, 'Model_Output_RandomForest.csv'), index=False)

    out_gb = df.copy()
    out_gb['Modelled_s0_GB']  = gb_preds
    out_gb['Residual']        = meas - gb_preds
    out_gb.to_csv(os.path.join(DIR, 'Model_Output_GradientBoosting.csv'), index=False)

    print(f"RF predictions  → Model_Output_RandomForest.csv")
    print(f"GB predictions  → Model_Output_GradientBoosting.csv")

    # ── Save stats ───────────────────────────────────────────────────────
    rf_rmse, rf_r, rf_cl = save_stats(
        meas, rf_preds, 'RandomForest', plots,
        os.path.join(DIR, 'Model_Stats_RandomForest.csv')
    )
    gb_rmse, gb_r, gb_cl = save_stats(
        meas, gb_preds, 'GradientBoosting', plots,
        os.path.join(DIR, 'Model_Stats_GradientBoosting.csv')
    )
    print(f"Stats saved → Model_Stats_RandomForest.csv / Model_Stats_GradientBoosting.csv")

    # ── Console summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  LOPO RESULTS SUMMARY  (n={n}, threshold={THRESHOLD} g/L)")
    print(f"{'='*60}")
    print(f"  {'Metric':<22} {'RandomForest':>14} {'GradBoost':>14}")
    print(f"  {'-'*50}")
    print(f"  {'RMSE (g/L)':<22} {rf_rmse:>14.5f} {gb_rmse:>14.5f}")
    print(f"  {'Pearson r':<22} {rf_r:>14.4f} {gb_r:>14.4f}")
    print(f"  {'Sensitivity':<22} {rf_cl['Sensitivity']*100:>13.1f}% {gb_cl['Sensitivity']*100:>13.1f}%")
    print(f"  {'Specificity':<22} {rf_cl['Specificity']*100:>13.1f}% {gb_cl['Specificity']*100:>13.1f}%")
    print(f"  {'F1':<22} {rf_cl['F1']:>14.4f} {gb_cl['F1']:>14.4f}")
    print(f"  {'Accuracy':<22} {rf_cl['Accuracy']*100:>13.1f}% {gb_cl['Accuracy']*100:>13.1f}%")
    print(f"{'='*60}")

    print(f"\nTop feature importances (full dataset):")
    for _, row in fi.iterrows():
        print(f"  {row['Feature']:<12}  RF={row['RF_Importance']*100:5.1f}%  "
              f"GB={row['GB_Importance']*100:5.1f}%")


if __name__ == '__main__':
    main()