"""
COMPARISON MODELS — RATING CURVE & RADIAL BASIS FUNCTION
=========================================================

RATING CURVE
------------
Power-law regression fit per-fold:  SSC = a · |U|^b
Represents the simplest hydraulic proxy — no spatial structure,
no canal topology, no vertical flux term.

RADIAL BASIS FUNCTION (RBF) INTERPOLATION
------------------------------------------
Gaussian RBF interpolation over [|U|, h, Δx] feature space.
Tests whether smooth spatial interpolation of the training SSC
field generalises to withheld canal configurations.

USAGE
-----
    python model_comparison.py

OUTPUTS
-------
    Model_Output_RatingCurve.csv        — LOPO predictions
    Model_Output_RBF.csv                — LOPO predictions
    Model_Stats_RatingCurve.csv         — performance metrics
    Model_Stats_RBF.csv                 — performance metrics
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import stats as sp
from scipy.interpolate import RBFInterpolator

# =============================================================================
# PATHS
# =============================================================================
DIR        = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(DIR, 'Input_Data.csv')

THRESHOLD  = 0.08


# =============================================================================
# HELPERS
# =============================================================================
def classification_stats(m, p, t=THRESHOLD):
    tp=int(((m>=t)&(p>=t)).sum()); fn=int(((m>=t)&(p<t)).sum())
    tn=int(((m<t)&(p<t)).sum());   fp=int(((m<t)&(p>=t)).sum())
    sens = tp/(tp+fn) if tp+fn>0 else 0.0
    spec = tn/(tn+fp) if tn+fp>0 else 0.0
    f1   = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn>0 else 0.0
    return dict(TP=tp,FN=fn,TN=tn,FP=fp,Sensitivity=sens,Specificity=spec,
                F1=f1,Accuracy=(tp+tn)/len(m))


def save_stats(m, p, label, out_path):
    res  = m - p
    n    = len(m)
    rmse = np.sqrt((res**2).mean())
    mae  = np.abs(res).mean()
    bias = res.mean()
    r_p, pv_p = sp.pearsonr(m, p)
    cl = classification_stats(m, p)
    rows = [
        ('Model',            label),
        ('n_obs',            n),
        ('RMSE_gL',          rmse),
        ('MAE_gL',           mae),
        ('Bias_gL',          bias),
        ('Pearson_r',        r_p),
        ('Pearson_r_pvalue', pv_p),
        ('Threshold_gL',     THRESHOLD),
        ('TP',               cl['TP']),
        ('FN',               cl['FN']),
        ('TN',               cl['TN']),
        ('FP',               cl['FP']),
        ('Sensitivity',      cl['Sensitivity']),
        ('Specificity',      cl['Specificity']),
        ('F1',               cl['F1']),
        ('Accuracy',         cl['Accuracy']),
    ]
    pd.DataFrame(rows, columns=['Metric','Value']).to_csv(out_path, index=False)
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
    U     = np.abs(df['Streamwise_Velocity_U_m_s'].values)
    h     = df['Water_Depth_m'].values
    dx    = df['Distance_to_Neighbor_m'].values

    unique_plots = np.unique(plots)

    rc_preds  = np.full(n, np.nan)
    rbf_preds = np.full(n, np.nan)

    print("Running LOPO cross-validation...\n")
    for plot in unique_plots:
        train = plots != plot
        test  = plots == plot
        print(f"  Withheld plot {plot}: train={train.sum()}  test={test.sum()}")

        # ── Rating curve: log-linear fit SSC ~ a * |U|^b ─────────────────
        U_tr   = U[train]; y_tr = meas[train]
        pos    = U_tr > 0
        if pos.sum() > 2:
            log_U = np.log(U_tr[pos])
            log_y = np.log(np.maximum(y_tr[pos], 1e-6))
            b_, log_a_, _, _, _ = sp.linregress(log_U, log_y)
            a_ = np.exp(log_a_)
            rc_preds[test] = np.maximum(a_ * U[test]**b_, 0.0)
        else:
            rc_preds[test] = np.mean(y_tr)

        # ── RBF interpolation over [|U|, h, Δx] ──────────────────────────
        X_tr  = np.column_stack([U[train], h[train], dx[train]])
        X_te  = np.column_stack([U[test],  h[test],  dx[test]])
        try:
            rbf = RBFInterpolator(X_tr, meas[train], kernel='gaussian',
                                  epsilon=1.0, smoothing=0.01)
            preds = rbf(X_te)
            rbf_preds[test] = np.maximum(preds, 0.0)
        except Exception as e:
            print(f"    RBF failed for plot {plot}: {e} — using mean fallback")
            rbf_preds[test] = np.mean(meas[train])

    # ── Save prediction CSVs ─────────────────────────────────────────────
    out_rc = df.copy()
    out_rc['Modelled_s0_RC'] = rc_preds
    out_rc['Residual']       = meas - rc_preds
    out_rc.to_csv(os.path.join(DIR, 'Model_Output_RatingCurve.csv'), index=False)

    out_rbf = df.copy()
    out_rbf['Modelled_s0_RBF'] = rbf_preds
    out_rbf['Residual']        = meas - rbf_preds
    out_rbf.to_csv(os.path.join(DIR, 'Model_Output_RBF.csv'), index=False)

    print(f"\nRC  predictions → Model_Output_RatingCurve.csv")
    print(f"RBF predictions → Model_Output_RBF.csv")

    # ── Save stats ───────────────────────────────────────────────────────
    rc_rmse, rc_r, rc_cl = save_stats(
        meas, rc_preds, 'RatingCurve',
        os.path.join(DIR, 'Model_Stats_RatingCurve.csv')
    )
    rbf_rmse, rbf_r, rbf_cl = save_stats(
        meas, rbf_preds, 'RBF',
        os.path.join(DIR, 'Model_Stats_RBF.csv')
    )
    print(f"Stats saved → Model_Stats_RatingCurve.csv / Model_Stats_RBF.csv")

    # ── Console summary ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  COMPARISON MODELS LOPO SUMMARY  (n={n})")
    print(f"{'='*55}")
    print(f"  {'Metric':<22} {'RatingCurve':>14} {'RBF':>10}")
    print(f"  {'-'*48}")
    print(f"  {'RMSE (g/L)':<22} {rc_rmse:>14.5f} {rbf_rmse:>10.5f}")
    print(f"  {'Pearson r':<22} {rc_r:>14.4f} {rbf_r:>10.4f}")
    print(f"  {'Sensitivity':<22} {rc_cl['Sensitivity']*100:>13.1f}% {rbf_cl['Sensitivity']*100:>9.1f}%")
    print(f"  {'Specificity':<22} {rc_cl['Specificity']*100:>13.1f}% {rbf_cl['Specificity']*100:>9.1f}%")
    print(f"  {'F1':<22} {rc_cl['F1']:>14.4f} {rbf_cl['F1']:>10.4f}")
    print(f"  {'Accuracy':<22} {rc_cl['Accuracy']*100:>13.1f}% {rbf_cl['Accuracy']*100:>9.1f}%")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()