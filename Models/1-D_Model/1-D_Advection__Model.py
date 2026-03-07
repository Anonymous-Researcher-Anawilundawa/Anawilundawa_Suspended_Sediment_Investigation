"""
SEDIMENT TRANSPORT MODEL — 1-D ADVECTION-FLUX
Geometry-Specific Diffusivity Multipliers
Optimized for LC, CPC Stem, and CPC Frond

Solves: u*h * ds/dx = -Fv
"""

import pandas as pd
import numpy as np
import os

# ========================================================================================
# 1. PARAMETERS  (all values match Table 4 of manuscript)
# ========================================================================================
KAPPA            = 0.44270    # von Karman constant (optimised v2)
REF_HEIGHT_RATIO = 0.76952    # Reference height ratio a/h (optimised v2)
DRAG_COEFF_CD    = 0.001899   # Drag coefficient (optimised v2)
DEFAULT_SETTLING_VEL = 0.0001186  # Settling velocity [m/s] (optimised v2)

# CALIBRATED DIFFUSIVITY MULTIPLIERS PER GEOMETRY (Table 4)
MULT_LC    = 1.1219  # Linear Canals — boundary roughness enhancement
MULT_STEM  = 0.8884  # CPC stem and junction locations
MULT_FROND = 1.3009  # CPC frond branches: Dean-number-driven secondary currents

WORKING_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE   = os.path.join(WORKING_DIR, "Input_Data.csv")
OUTPUT_FILE  = os.path.join(WORKING_DIR, "Model_Output_No_Diffusion.csv")
REPORT_FILE  = os.path.join(WORKING_DIR, "Model_Validation_and_Stability_Report.md")

# ========================================================================================
# 2. CORE PHYSICS ENGINE
# ========================================================================================

def get_geometry_multiplier(row):
    """Assigns diffusivity multiplier based on canal geometry and location code."""
    loc   = row['Location']
    ctype = row['Canal_Type']
    if ctype == 'LC':
        return MULT_LC
    if loc in ['E', 'G']:       # CPC frond tips
        return MULT_FROND
    return MULT_STEM             # CPC stem, junctions, and remaining locations


def solve_marching(s_prev, dx, u, h, sb, eps_z, mult):
    """
    Advances concentration one node downstream.

    Implements Equation 7:  s_next = s_prev - (Fv * dx) / (u * h)

    where Fv = F_settling - F_diffusive  (Equation 3)
    F_settling  = omega * sb             (Equation 4)
    F_diffusive = eps_z * P * sb / (h-a) (Equation 6)
    """
    u_star = max(abs(u), 1e-6) * np.sqrt(DRAG_COEFF_CD / 2.0)
    P      = DEFAULT_SETTLING_VEL / (KAPPA * u_star)
    a      = REF_HEIGHT_RATIO * h

    f_set  = DEFAULT_SETTLING_VEL * sb
    f_diff = (eps_z * mult) * (P * sb) / (h - a)
    Fv     = f_set - f_diff

    delta_s = -(Fv * dx) / (abs(u) * h)
    return max(s_prev + delta_s, 0.0)


# ========================================================================================
# 3. MAIN EXECUTION
# ========================================================================================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    df['Modelled_s0']  = 0.0
    df['Is_Seed_Node'] = False   # flag for inlet/boundary nodes
    results_map = {}

    # Topological sort: upstream nodes processed before downstream
    df.sort_values(
        by=['Plot', 'Sampling_Day', 'Tide_Phase', 'Distance_to_Neighbor_m'],
        inplace=True
    )

    print("Executing geometry-aware 1-D advection model...")

    for index, row in df.iterrows():
        u      = row['Streamwise_Velocity_U_m_s']
        h      = row['Water_Depth_m']
        sb     = row['Input_Near_Bed_Conc_sb_g_L']
        eps_z  = row['Diffusivity_Vertical_m2_s']
        dx     = row['Distance_to_Neighbor_m']
        target = row['Target_Measured_Avg_s0_g_L']
        mult   = get_geometry_multiplier(row)

        current_key = (f"{row['Plot']}_{row['Location']}_"
                       f"{row['Sampling_Day']}_{row['Tide_State']}")
        parent_key  = (f"{row['Plot']}_{row['Neighbor_Node']}_"
                       f"{row['Sampling_Day']}_{row['Tide_State']}")

        # Seed nodes: inlet boundaries seeded with measured SSC
        if dx == 0 or row['Neighbor_Node'] == "Dutch_Canal":
            s_mod = target
            df.at[index, 'Is_Seed_Node'] = True
        else:
            s_prev = results_map.get(parent_key, target)
            s_mod  = solve_marching(s_prev, dx, u, h, sb, eps_z, mult)

        results_map[current_key] = s_mod
        df.at[index, 'Modelled_s0'] = s_mod

    # ========================================================================================
    # 4. VALIDATION STATISTICS — seed nodes excluded (s_mod == target by definition there)
    # ========================================================================================
    val   = df[df['Is_Seed_Node'] == False].copy()
    meas  = val['Target_Measured_Avg_s0_g_L']
    mod   = val['Modelled_s0']
    res   = meas - mod
    rmse  = np.sqrt((res ** 2).mean())
    mae   = res.abs().mean()
    bias  = res.mean()
    r2    = 1 - (np.sum(res ** 2) / np.sum((meas - meas.mean()) ** 2))

    print(f"\nValidation Statistics (seed nodes excluded, n={len(val)}):")
    print(f"  RMSE : {rmse:.5f} g/L")
    print(f"  MAE  : {mae:.5f} g/L")
    print(f"  Bias : {bias:.5f} g/L")
    print(f"  R²   : {r2:.4f}")

    df.to_csv(OUTPUT_FILE, index=False)
    with open(REPORT_FILE, "w") as f:
        f.write("# 1-D GEOMETRY-OPTIMISED MODEL — VALIDATION REPORT\n\n")
        f.write(f"Seed nodes excluded from statistics (n validated = {len(val)})\n\n")
        f.write(f"RMSE : {rmse:.5f} g/L\n")
        f.write(f"MAE  : {mae:.5f} g/L\n")
        f.write(f"Bias : {bias:.5f} g/L\n")
        f.write(f"R²   : {r2:.4f}\n")

    print(f"\nOutput saved to : {OUTPUT_FILE}")
    print(f"Report saved to : {REPORT_FILE}")


if __name__ == "__main__":
    main()