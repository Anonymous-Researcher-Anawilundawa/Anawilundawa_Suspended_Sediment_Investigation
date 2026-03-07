"""
EXPLICIT FOCDMS — FIRST-ORDER CENTRED-DIFFERENCE MARCHING SCHEME
=================================================================
Solves: u * ds/dx  =  D * d²s/dx²  -  Fv/h

Discretisation: centred-space for diffusion (explicit, forward-marching).

PRE-OPTIMISATION PARAMETERS are used intentionally. This model is a
comparison case demonstrating structural failure due to Pe >> 2, not a
calibrated predictive tool. The physical diffusivity D = 1.5×10⁻⁵ m²/s
is used for the Pe stability argument.

USAGE
-----
    python model_focdms.py

OUTPUTS
-------
    Model_Output_FOCDMS.csv     — per-node predictions and Pe numbers
    Model_Stats_FOCDMS.csv      — performance and stability metrics
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import stats as sp

# =============================================================================
# PATHS
# =============================================================================
DIR        = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(DIR, 'Input_Data.csv')
OUTPUT_CSV = os.path.join(DIR, 'Model_Output_FOCDMS.csv')
STATS_CSV  = os.path.join(DIR, 'Model_Stats_FOCDMS.csv')

# =============================================================================
# PRE-OPTIMISATION PARAMETERS (intentional — see module docstring)
# =============================================================================
KAPPA       = 0.41
REF_H_RATIO = 0.75
C_D         = 0.0025
OMEGA       = 0.0005    # settling velocity [m/s]
D_PHYS      = 1.5e-5   # physical molecular diffusivity [m²/s]

THRESHOLD   = 0.08
SMALL       = 1e-9


# =============================================================================
# PHYSICAL FUNCTIONS
# =============================================================================
def vertical_flux(U, h, sb, eps_z):
    u_star = max(abs(U), SMALL) * np.sqrt(C_D / 2.0)
    P      = OMEGA / max(KAPPA * u_star, SMALL)
    a      = REF_H_RATIO * h
    return OMEGA * sb - eps_z * (P * sb) / max(h - a, SMALL)


def focdms_step(s_prev, s_upstream, Fv, dx, U, h, D):
    """
    Explicit centred-space advection-diffusion step.
    d²s/dx² approximated as backward first difference (using upstream node).
    """
    U_eff  = U if abs(U) > SMALL else SMALL
    d2s    = (s_prev - s_upstream) / dx
    ds_dx  = -(Fv / (U_eff * h)) + (D * d2s) / (U_eff * h)
    return max(s_prev + ds_dx * dx, 0.0)


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not os.path.isfile(INPUT_FILE):
        sys.exit(f"ERROR: Cannot find {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    df = df.reset_index(drop=True)
    df['Modelled_s0']   = np.nan
    df['Peclet_Number'] = np.nan
    df['Is_Seed_Node']  = False

    results_map  = {}
    upstream_map = {}

    df_sorted = df.sort_values(
        ['Plot', 'Sampling_Day', 'Tide_Phase', 'Distance_to_Neighbor_m']
    ).copy()

    def ck(row): return f"{row['Plot']}_{row['Location']}_{row['Sampling_Day']}_{row['Tide_State']}"
    def pk(row): return f"{row['Plot']}_{row['Neighbor_Node']}_{row['Sampling_Day']}_{row['Tide_State']}"

    for idx, row in df_sorted.iterrows():
        U    = row['Streamwise_Velocity_U_m_s']
        h    = row['Water_Depth_m']
        sb   = row['Input_Near_Bed_Conc_sb_g_L']
        eps_z= row['Diffusivity_Vertical_m2_s']
        dx   = row['Distance_to_Neighbor_m']
        tgt  = row['Target_Measured_Avg_s0_g_L']
        Pe   = abs(U) * dx / D_PHYS if dx > 0 else 0.0

        df.at[idx, 'Peclet_Number'] = Pe

        k = ck(row); p = pk(row)
        Fv = vertical_flux(U, h, sb, eps_z)

        if dx == 0 or row['Neighbor_Node'] == 'Dutch_Canal':
            s0 = tgt
            df.at[idx, 'Is_Seed_Node'] = True
            upstream_map[k] = tgt
        else:
            s_prev = results_map.get(p, tgt)
            s_up   = upstream_map.get(p, s_prev)
            s0     = focdms_step(s_prev, s_up, Fv, dx, U, h, D_PHYS)
            upstream_map[k] = s_prev

        results_map[k] = s0
        df.at[idx, 'Modelled_s0'] = s0

    df['Residual']      = df['Target_Measured_Avg_s0_g_L'] - df['Modelled_s0']
    df['Abs_Error']     = df['Residual'].abs()
    df['Squared_Error'] = df['Residual'] ** 2
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved → {OUTPUT_CSV}")

    # ── Statistics ──────────────────────────────────────────────────────────
    val = df[df['Is_Seed_Node'] == False].copy()
    m   = val['Target_Measured_Avg_s0_g_L'].values
    p_  = val['Modelled_s0'].values
    pe  = val['Peclet_Number'].values
    res = m - p_
    n   = len(m)

    rmse = np.sqrt((res**2).mean())
    mae  = np.abs(res).mean()
    bias = res.mean()
    r_p, pv_p = sp.pearsonr(m, p_)

    def cls(m_, p__, t=THRESHOLD):
        tp=int(((m_>=t)&(p__>=t)).sum()); fn=int(((m_>=t)&(p__<t)).sum())
        tn=int(((m_<t)&(p__<t)).sum());   fp=int(((m_<t)&(p__>=t)).sum())
        sens=tp/(tp+fn) if tp+fn>0 else 0.0
        spec=tn/(tn+fp) if tn+fp>0 else 0.0
        f1  =2*tp/(2*tp+fp+fn) if 2*tp+fp+fn>0 else 0.0
        return dict(TP=tp,FN=fn,TN=tn,FP=fp,Sensitivity=sens,Specificity=spec,
                    F1=f1,Accuracy=(tp+tn)/len(m_))

    cl = cls(m, p_)

    stats_rows = [
        ('n_obs',               n),
        ('RMSE_gL',             rmse),
        ('MAE_gL',              mae),
        ('Bias_gL',             bias),
        ('Pearson_r',           r_p),
        ('Pearson_r_pvalue',    pv_p),
        ('Pe_mean',             pe[pe>0].mean()),
        ('Pe_median',           np.median(pe[pe>0])),
        ('Pe_min',              pe[pe>0].min()),
        ('Pe_max',              pe.max()),
        ('Pct_Pe_gt2',          (pe>2).mean()*100),
        ('D_phys_m2s',          D_PHYS),
        ('Threshold_gL',        THRESHOLD),
        ('TP',                  cl['TP']),
        ('FN',                  cl['FN']),
        ('TN',                  cl['TN']),
        ('FP',                  cl['FP']),
        ('Sensitivity',         cl['Sensitivity']),
        ('Specificity',         cl['Specificity']),
        ('F1',                  cl['F1']),
        ('Accuracy',            cl['Accuracy']),
    ]
    pd.DataFrame(stats_rows, columns=['Metric','Value']).to_csv(STATS_CSV, index=False)
    print(f"Stats saved     → {STATS_CSV}")

    print(f"\n{'='*55}")
    print(f"  FOCDMS EXPLICIT — RESULTS SUMMARY  (n={n})")
    print(f"{'='*55}")
    print(f"  RMSE        : {rmse:.5f} g/L")
    print(f"  Pearson r   : {r_p:.4f}")
    print(f"  Sensitivity : {cl['Sensitivity']*100:.1f}%  TP={cl['TP']} FN={cl['FN']}")
    print(f"  Specificity : {cl['Specificity']*100:.1f}%  TN={cl['TN']} FP={cl['FP']}")
    print(f"  F1          : {cl['F1']:.4f}")
    print(f"  Pe mean={pe[pe>0].mean():,.0f}  median={np.median(pe[pe>0]):,.0f}  max={pe.max():,.0f}")
    print(f"  % Pe > 2    : {(pe>2).mean()*100:.1f}%")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()