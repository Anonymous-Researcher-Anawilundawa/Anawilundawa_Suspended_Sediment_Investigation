"""
IMPLICIT TDMA ADVECTION-DIFFUSION SOLVER
=========================================
Solves steady 1-D advection-diffusion-reaction for each canal chain:

    u * ds/dx  =  D * d²s/dx²  -  Fv/h

Discretisation:
    Advection  : first-order upwind  (unconditionally stable for any Pe)
    Diffusion  : second-order central, fully implicit
    Reaction   : vertical flux Fv/h as distributed sink

Boundary conditions:
    Inlet    : Dirichlet — measured SSC
    Terminus : Neumann   — zero gradient (ghost-node implementation)

OUTPUTS
-------
    Model_Output_TDMA.csv     — per-node predictions
    Model_Stats_TDMA.csv      — performance and diffusivity metrics
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
OUTPUT_CSV = os.path.join(DIR, 'Model_Output_TDMA.csv')
STATS_CSV  = os.path.join(DIR, 'Model_Stats_TDMA.csv')

# =============================================================================
# PRE-OPTIMISATION PARAMETERS (intentional — see module docstring)
# =============================================================================
KAPPA       = 0.41
REF_H_RATIO = 0.75
C_D         = 0.0025
OMEGA       = 0.0005
D_PHYS      = 1.5e-5   # physical molecular diffusivity [m²/s]

THRESHOLD   = 0.08
SMALL       = 1e-30

# Network chains
CPC_CHAINS = [
    ['C', 'D'],
    ['D', 'E'],
    ['D', 'F', 'G'],
    ['D', 'F_across', 'H'],
]
LC_CHAIN = ['I', 'J', 'K']


# =============================================================================
# PHYSICAL FUNCTIONS
# =============================================================================
def vertical_flux(U, h, sb, eps_z):
    u_star = max(abs(U), 1e-9) * np.sqrt(C_D / 2.0)
    P      = OMEGA / max(KAPPA * u_star, 1e-9)
    a      = REF_H_RATIO * h
    return OMEGA * sb - eps_z * (P * sb) / max(h - a, 1e-9)


# =============================================================================
# THOMAS ALGORITHM (TDMA)
# =============================================================================
def tdma_solve(a, b, c, d):
    n  = len(b)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x  = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i-1]
        denom = denom if abs(denom) > SMALL else SMALL
        cp[i] = (c[i] / denom) if i < n-1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i-1]) / denom
    x[n-1] = dp[n-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x


def solve_chain(node_params, s_inlet):
    """
    Assemble and solve tridiagonal system for one canal chain.
    node_params: list of dicts with keys U, h, Fv, dx (distance from prev node)
    """
    n = len(node_params)
    if n == 1:
        return np.array([s_inlet])

    a_ = np.zeros(n); b_ = np.zeros(n)
    c_ = np.zeros(n); d_ = np.zeros(n)

    # Node 0 — Dirichlet
    b_[0] = 1.0; d_[0] = s_inlet

    # Interior nodes
    for i in range(1, n-1):
        nd   = node_params[i]
        U, h, Fv = nd['U'], nd['h'], nd['Fv']
        dxL  = nd['dx']
        dxR  = node_params[i+1]['dx']
        gL   = 1.0 / dxL; gR = 1.0 / dxR
        dm   = (dxL + dxR) / 2.0

        if U >= 0:
            adv_m1 = -U/dxL; adv_0 =  U/dxL; adv_p1 = 0.0
        else:
            adv_m1 =  0.0;   adv_0 = -U/dxR; adv_p1 = U/dxR

        a_[i] = adv_m1 + D_PHYS * gL / dm
        b_[i] = adv_0  + D_PHYS * (gL + gR) / dm
        c_[i] = adv_p1 - D_PHYS * gR / dm
        d_[i] = -Fv / h

    # Last node — Neumann (ghost node: s[n] = s[n-2])
    nd   = node_params[-1]
    U, h, Fv, dxL = nd['U'], nd['h'], nd['Fv'], nd['dx']
    coef = abs(U) / dxL + 2.0 * D_PHYS / dxL**2
    a_[-1] = -coef; b_[-1] = coef; d_[-1] = -Fv / h

    s = tdma_solve(a_, b_, c_, d_)
    return np.maximum(s, 0.0)


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not os.path.isfile(INPUT_FILE):
        sys.exit(f"ERROR: Cannot find {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE).reset_index(drop=True)
    df['Modelled_s0_implicit'] = np.nan
    df['Peclet_Number']        = np.nan
    df['Is_Seed_Node']         = False

    for (plot, day, tide), grp in df.groupby(['Plot', 'Sampling_Day', 'Tide_State']):
        grp     = grp.sort_values('Distance_to_Neighbor_m')
        canal   = grp['Canal_Type'].iloc[0]
        chains  = CPC_CHAINS if canal == 'CPC' else [LC_CHAIN]

        for chain in chains:
            nodes  = []
            idxs   = []
            for loc in chain:
                row = grp[grp['Location'] == loc]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                Fv = vertical_flux(
                    r['Streamwise_Velocity_U_m_s'],
                    r['Water_Depth_m'],
                    r['Input_Near_Bed_Conc_sb_g_L'],
                    r['Diffusivity_Vertical_m2_s']
                )
                nodes.append({
                    'U':  r['Streamwise_Velocity_U_m_s'],
                    'h':  r['Water_Depth_m'],
                    'Fv': Fv,
                    'dx': r['Distance_to_Neighbor_m'],
                })
                idxs.append(row.index[0])

            if len(nodes) < 2:
                continue

            # First node is inlet (Dirichlet)
            df.at[idxs[0], 'Is_Seed_Node'] = True
            s_inlet = df.at[idxs[0], 'Target_Measured_Avg_s0_g_L']
            result  = solve_chain(nodes, s_inlet)

            for ii, idx in enumerate(idxs):
                df.at[idx, 'Modelled_s0_implicit'] = result[ii]

    # Péclet numbers
    df['Peclet_Number'] = (
        np.abs(df['Streamwise_Velocity_U_m_s']) *
        df['Distance_to_Neighbor_m'] / D_PHYS
    )

    df['Residual']      = df['Target_Measured_Avg_s0_g_L'] - df['Modelled_s0_implicit']
    df['Abs_Error']     = df['Residual'].abs()
    df['Squared_Error'] = df['Residual'] ** 2
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved → {OUTPUT_CSV}")

    # ── Statistics ──────────────────────────────────────────────────────────
    val   = df[df['Is_Seed_Node'] == False].copy()
    valid = val['Modelled_s0_implicit'].notna()
    m     = val.loc[valid, 'Target_Measured_Avg_s0_g_L'].values
    p_    = val.loc[valid, 'Modelled_s0_implicit'].values.astype(float)
    pe    = val['Peclet_Number'].values
    res   = m - p_
    n     = len(m)

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

    # Effective diffusivity ratio
    seed = (df['Distance_to_Neighbor_m']==0)|(df['Neighbor_Node']=='Dutch_Canal')
    U_ns = np.abs(df.loc[~seed, 'Streamwise_Velocity_U_m_s'].values)
    dx_ns = df.loc[~seed, 'Distance_to_Neighbor_m'].values
    D_eff_ratio = 0.5 * U_ns * dx_ns / D_PHYS

    stats_rows = [
        ('n_valid',             n),
        ('n_NaN',               int(~valid.sum())),
        ('RMSE_gL',             rmse),
        ('MAE_gL',              mae),
        ('Bias_gL',             bias),
        ('Pearson_r',           r_p),
        ('Pearson_r_pvalue',    pv_p),
        ('Pe_mean',             pe[pe>0].mean()),
        ('Pe_median',           np.median(pe[pe>0])),
        ('Pe_max',              pe.max()),
        ('Pct_Pe_gt2',          (pe>2).mean()*100),
        ('D_eff_ratio_mean',    D_eff_ratio.mean()),
        ('D_eff_ratio_median',  np.median(D_eff_ratio)),
        ('D_eff_ratio_max',     D_eff_ratio.max()),
        ('Pct_Deff_gt1000x',    (D_eff_ratio>1000).mean()*100),
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
    print(f"  TDMA IMPLICIT — RESULTS SUMMARY  (n_valid={n})")
    print(f"{'='*55}")
    print(f"  RMSE         : {rmse:.5f} g/L")
    print(f"  Pearson r    : {r_p:.4f}")
    print(f"  Sensitivity  : {cl['Sensitivity']*100:.1f}%  TP={cl['TP']} FN={cl['FN']}")
    print(f"  Specificity  : {cl['Specificity']*100:.1f}%  TN={cl['TN']} FP={cl['FP']}")
    print(f"  F1           : {cl['F1']:.4f}")
    print(f"  D_eff/D_phys : mean={D_eff_ratio.mean():,.0f}x  median={np.median(D_eff_ratio):,.0f}x")
    print(f"  Pe mean      : {pe[pe>0].mean():,.0f}")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()