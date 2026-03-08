"""
1-D UPWIND ADVECTION MODEL
===========================
Solves steady 1-D advection with vertical settling flux for each canal chain:

    u * ds/dx  =  -Fv / h

Discretisation  : first-order upwind advection (no longitudinal diffusion)
Network topology: CPC chains [C→D], [D→E], [D→F→G], [D→F_across→H]
                  LC chain   [I→J→K]
Boundary cond.  : Dirichlet at inlet (measured SSC); Neumann at terminus

CALIBRATED PARAMETERS (optimised v2, differential evolution + Nelder-Mead,
values match Table 3 of manuscript):
    kappa            = 0.39955
    ref_height_ratio = 0.79477
    C_D              = 0.002690
    omega            = 0.000103   m/s
    mult_LC          = 1.0981     (geometry-specific diffusivity multiplier)
    mult_stem        = 0.9868     (geometry-specific diffusivity multiplier)
    mult_frond_LT    = 1.5106     (low tide frond diffusivity multiplier)
    mult_frond_HT    = 1.4533     (high tide frond diffusivity multiplier)

OUTPUTS
-------
    Model_Output_1D_Upwind.csv          — per-node predictions + residuals
    Model_Stats_1D_Upwind.csv           — continuous and classification metrics
    Model_Stats_Splits_1D_Upwind.csv    — tidal state / canal type breakdown
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats as sp

# =============================================================================
# PATHS  (script-relative — place Input_Data.csv alongside this file)
# =============================================================================
DIR        = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(DIR, 'Input_Data.csv')
OUTPUT_CSV = os.path.join(DIR, 'Model_Output_1D_Upwind.csv')
STATS_CSV  = os.path.join(DIR, 'Model_Stats_1D_Upwind.csv')
SPLITS_CSV = os.path.join(DIR, 'Model_Stats_Splits_1D_Upwind.csv')

# =============================================================================
# CALIBRATED PARAMETERS  (Table 3 of manuscript)
# =============================================================================
KAPPA         = 0.39955
REF_H_RATIO   = 0.79477
C_D           = 0.002690
OMEGA         = 0.000103    # settling velocity [m/s]
MULT_LC       = 1.0981      # Linear Canal diffusivity multiplier
MULT_STEM     = 0.9868      # CPC stem diffusivity multiplier
MULT_FROND_LT = 1.5106      # CPC frond diffusivity multiplier — low tide
MULT_FROND_HT = 1.4533      # CPC frond diffusivity multiplier — high tide

THRESHOLD = 0.08            # ecological SSC dieback threshold [g/L]
SMALL     = 1e-9            # numerical floor to prevent division by zero


# =============================================================================
# DIFFUSIVITY MULTIPLIER ASSIGNMENT
# =============================================================================
def diffusivity_multiplier(canal_type, location, tide_state):
    """
    Returns the geometry- and tide-specific diffusivity multiplier for a node.

    LC canals: uniform enhancement from boundary roughness.
    CPC canals:
      - Stem nodes (C, D): mild suppression.
      - Frond nodes (E, G): Dean-number-driven secondary currents enhance
        vertical mixing; split by tidal state.
      - All other CPC nodes (F, F_across, H): treated as stem.
    """
    if canal_type == 'LC':
        return MULT_LC
    if location in ('E', 'G'):
        return MULT_FROND_LT if tide_state == 'LT' else MULT_FROND_HT
    return MULT_STEM


# =============================================================================
# PHYSICS: VERTICAL FLUX
# =============================================================================
def vertical_flux(U, h, sb, eps_z, mult):
    """
    Computes net vertical SSC flux Fv = F_settling - F_diffusive  [g/L · m/s].

    Implements Equations 3–6 of the manuscript:
      F_settling  = omega * sb                         (Eq. 4)
      F_diffusive = (eps_z * mult) * P * sb / (h - a)  (Eq. 6)
      Fv          = F_settling - F_diffusive            (Eq. 3)

    The diffusivity multiplier (mult) scales eps_z only — it is NOT applied
    to the advective velocity U.

    Parameters
    ----------
    U     : streamwise velocity [m/s]
    h     : water depth [m]
    sb    : near-bed SSC [g/L]
    eps_z : vertical diffusivity [m²/s]
    mult  : geometry-specific diffusivity multiplier [-]
    """
    u_star = max(abs(U), SMALL) * np.sqrt(C_D / 2.0)
    P      = OMEGA / max(KAPPA * u_star, SMALL)       # Rouse number
    a      = REF_H_RATIO * h                           # reference height
    Fv     = OMEGA * sb - (eps_z * mult) * (P * sb) / max(h - a, SMALL)
    return Fv


# =============================================================================
# KEY HELPERS
# =============================================================================
def node_key(row):
    return f"{row['Plot']}_{row['Location']}_{row['Sampling_Day']}_{row['Tide_State']}"

def parent_key(row):
    return f"{row['Plot']}_{row['Neighbor_Node']}_{row['Sampling_Day']}_{row['Tide_State']}"


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================
def classification_metrics(measured, modelled, threshold=THRESHOLD):
    """Returns TP/FN/TN/FP and derived metrics at a given SSC threshold."""
    m = np.asarray(measured)
    p = np.asarray(modelled)
    TP = int(((m >= threshold) & (p >= threshold)).sum())
    FN = int(((m >= threshold) & (p <  threshold)).sum())
    TN = int(((m <  threshold) & (p <  threshold)).sum())
    FP = int(((m <  threshold) & (p >= threshold)).sum())
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1          = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    accuracy    = (TP + TN) / len(m)
    return dict(TP=TP, FN=FN, TN=TN, FP=FP,
                Sensitivity=sensitivity, Specificity=specificity,
                F1=f1, Accuracy=accuracy)


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not os.path.isfile(INPUT_FILE):
        sys.exit(f"ERROR: Input file not found — {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE).reset_index(drop=True)
    df['Modelled_s0']  = np.nan
    df['Is_Seed_Node'] = False

    results_map = {}   # node_key → modelled SSC

    # Process nodes in upstream-to-downstream order
    df_sorted = df.sort_values(
        ['Plot', 'Sampling_Day', 'Tide_Phase', 'Distance_to_Neighbor_m']
    ).copy()

    for idx, row in df_sorted.iterrows():
        U     = row['Streamwise_Velocity_U_m_s']
        h     = row['Water_Depth_m']
        sb    = row['Input_Near_Bed_Conc_sb_g_L']
        eps_z = row['Diffusivity_Vertical_m2_s']
        dx    = row['Distance_to_Neighbor_m']
        tgt   = row['Target_Measured_Avg_s0_g_L']

        mult  = diffusivity_multiplier(row['Canal_Type'], row['Location'],
                                       row['Tide_State'])
        ck    = node_key(row)
        pk    = parent_key(row)

        if dx == 0 or row['Neighbor_Node'] == 'Dutch_Canal':
            # Inlet / seed node — Dirichlet boundary condition
            s0 = tgt
            df.at[idx, 'Is_Seed_Node'] = True
        else:
            s_prev = results_map.get(pk, tgt)
            Fv     = vertical_flux(U, h, sb, eps_z, mult)
            # Upwind advection step (Eq. 7):  s_next = s_prev - (Fv * dx) / (U * h)
            U_eff  = abs(U) if abs(U) > SMALL else SMALL
            s0     = max(s_prev - (Fv * dx) / (U_eff * h), 0.0)

        results_map[ck]        = s0
        df.at[idx, 'Modelled_s0'] = s0

    # ── Residuals ────────────────────────────────────────────────────────────
    df['Residual']      = df['Target_Measured_Avg_s0_g_L'] - df['Modelled_s0']
    df['Abs_Error']     = df['Residual'].abs()
    df['Squared_Error'] = df['Residual'] ** 2
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved  →  {OUTPUT_CSV}")

    # ── Validation statistics (seed nodes excluded) ───────────────────────
    val = df[df['Is_Seed_Node'] == False].copy()
    m   = val['Target_Measured_Avg_s0_g_L'].values
    p   = val['Modelled_s0'].values
    res = m - p
    n   = len(m)

    rmse              = np.sqrt((res ** 2).mean())
    mae               = np.abs(res).mean()
    bias              = res.mean()                         # mean(measured − modelled)
    r_p,  pv_p        = sp.pearsonr(m, p)
    r_s,  pv_s        = sp.spearmanr(m, p)
    t_stat, t_p       = sp.ttest_rel(m, p)
    w_stat, w_p       = sp.wilcoxon(res)
    sw_w,   sw_p      = sp.shapiro(res)

    # Bland-Altman limits of agreement
    ba_lo      = bias - 1.96 * res.std()
    ba_hi      = bias + 1.96 * res.std()
    within_ba  = ((res >= ba_lo) & (res <= ba_hi)).mean() * 100

    # Linear regression — full dataset and outliers removed (>0.35 g/L)
    sl_full, ic_full, r_full, _, _ = sp.linregress(m, p)
    no_out = (m <= 0.35) & (p <= 0.35)
    sl_no,  ic_no,  r_no,  _, _ = sp.linregress(m[no_out], p[no_out])

    # Bootstrap prediction interval width (1 000 resamples)
    rng     = np.random.RandomState(42)
    boot_pi = []
    for _ in range(1000):
        idx_   = rng.choice(n, n, replace=True)
        r_b    = res[idx_]
        boot_pi.append(np.percentile(r_b, 97.5) - np.percentile(r_b, 2.5))
    pi_mean = np.mean(boot_pi)

    # Classification at ecological threshold
    cl = classification_metrics(m, p)

    # ── Subgroup splits ───────────────────────────────────────────────────
    ht  = (val['Tide_State']  == 'HT').values
    lt  = (val['Tide_State']  == 'LT').values
    cpc = (val['Canal_Type']  == 'CPC').values
    lc  = (val['Canal_Type']  == 'LC').values
    splits = {}
    for label, mask in [('HT', ht), ('LT', lt), ('CPC', cpc), ('LC', lc)]:
        splits[f'{label}_n']    = int(mask.sum())
        splits[f'{label}_RMSE'] = np.sqrt(((m[mask] - p[mask]) ** 2).mean())
        for k, v in classification_metrics(m[mask], p[mask]).items():
            splits[f'{label}_{k}'] = v

    # ── Save statistics ───────────────────────────────────────────────────
    stats_rows = [
        ('n_obs',                     n),
        ('RMSE_gL',                   rmse),
        ('MAE_gL',                    mae),
        ('Bias_gL',                   bias),
        ('BA_lower_gL',               ba_lo),
        ('BA_upper_gL',               ba_hi),
        ('Within_BA_pct',             within_ba),
        ('Pearson_r',                 r_p),
        ('Pearson_r_pvalue',          pv_p),
        ('Spearman_rho',              r_s),
        ('Spearman_rho_pvalue',       pv_s),
        ('PairedT_t',                 t_stat),
        ('PairedT_df',                n - 1),
        ('PairedT_pvalue',            t_p),
        ('Wilcoxon_W',                w_stat),
        ('Wilcoxon_pvalue',           w_p),
        ('ShapiroWilk_W',             sw_w),
        ('ShapiroWilk_pvalue',        sw_p),
        ('Regression_slope_full',     sl_full),
        ('Regression_intercept_full', ic_full),
        ('Regression_r_full',         r_full),
        ('Regression_slope_noout',    sl_no),
        ('Regression_intercept_noout',ic_no),
        ('Regression_r_noout',        r_no),
        ('n_noout',                   int(no_out.sum())),
        ('Bootstrap_PI_mean_width',   pi_mean),
        ('Threshold_gL',              THRESHOLD),
        ('TP',                        cl['TP']),
        ('FN',                        cl['FN']),
        ('TN',                        cl['TN']),
        ('FP',                        cl['FP']),
        ('Sensitivity',               cl['Sensitivity']),
        ('Specificity',               cl['Specificity']),
        ('F1',                        cl['F1']),
        ('Accuracy',                  cl['Accuracy']),
    ]
    pd.DataFrame(stats_rows, columns=['Metric', 'Value']).to_csv(STATS_CSV,  index=False)
    pd.DataFrame([splits]).to_csv(SPLITS_CSV, index=False)
    print(f"Stats saved        →  {STATS_CSV}")
    print(f"Split stats saved  →  {SPLITS_CSV}")

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n{'='*57}")
    print(f"  1-D UPWIND MODEL — VALIDATION SUMMARY  (n = {n})")
    print(f"{'='*57}")
    print(f"  RMSE          : {rmse:.5f} g/L")
    print(f"  MAE           : {mae:.5f} g/L")
    print(f"  Bias          : {bias:+.5f} g/L  (neg = model overestimates)")
    print(f"  Pearson r     : {r_p:.4f}   (p = {pv_p:.4f})")
    print(f"  Spearman rho  : {r_s:.4f}   (p = {pv_s:.4f})")
    print(f"  Paired t      : t = {t_stat:.3f},  p = {t_p:.4f}")
    print(f"  Wilcoxon W    : {w_stat:.0f}   (p = {w_p:.4f})")
    print(f"  Shapiro-Wilk  : W = {sw_w:.3f},  p = {sw_p:.4f}")
    print(f"  BA limits     : {ba_lo:.4f} to {ba_hi:.4f}  ({within_ba:.1f}% within)")
    print(f"  Bootstrap PI  : {pi_mean:.4f} g/L (mean width, 1 000 resamples)")
    print(f"  Regression    : slope = {sl_full:.3f},  intercept = {ic_full:.4f}  (full)")
    print(f"                  slope = {sl_no:.3f},  intercept = {ic_no:.4f}  (outliers removed)")
    print(f"  Sensitivity   : {cl['Sensitivity']*100:.1f}%   TP = {cl['TP']},  FN = {cl['FN']}")
    print(f"  Specificity   : {cl['Specificity']*100:.1f}%   TN = {cl['TN']},  FP = {cl['FP']}")
    print(f"  F1 score      : {cl['F1']:.4f}")
    print(f"  Accuracy      : {cl['Accuracy']*100:.1f}%")
    print(f"{'='*57}")


if __name__ == '__main__':
    main()