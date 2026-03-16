"""
Stylised loan portfolio construction — LTV-based structural model.

Primary state variables are LTV₀ (loan-to-value at origination) and PV₀
(property value at origination). All credit risk characteristics are derived
structurally from these, following the standard mortgage credit risk literature:

  PD_12m  = logistic(β₀ + β₁ × LTV₀)                     [Ferretti et al. 2023]
  LGD     = max(1 − (1−ω) / LTV₀, 0)  at d=0              [Qi & Yang 2009]
  EAD     = LTV₀ × PV₀                                     [structural identity]

A reduced-form portfolio (independent EAD, PD, LGD draws) is also generated as
additional columns for use in the sensitivity model (Notebook 05).
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize


# ---------------------------------------------------------------------------
# Credit risk helper functions (LTV structural model)
# ---------------------------------------------------------------------------

def _logistic_pd(ltv: np.ndarray, beta_0: float, beta_1: float) -> np.ndarray:
    """12-month PD as logistic function of current LTV."""
    return 1.0 / (1.0 + np.exp(-(beta_0 + beta_1 * ltv)))


def _collateral_shortfall_lgd(
    ltv_0: np.ndarray, d: float, omega: float
) -> np.ndarray:
    """
    LGD from collateral shortfall net of workout costs.

    LGD(d) = max(1 − (1−d)(1−ω) / LTV₀, 0)

    At d=0 (baseline): LGD = max(1 − (1−ω)/LTV₀, 0).
    Loans with LTV₀ < (1−ω) have zero baseline LGD — the collateral fully covers
    the debt even after workout discounts.
    """
    return np.maximum(1.0 - (1.0 - d) * (1.0 - omega) / ltv_0, 0.0)


def _lifetime_pd(pd_12m: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Lifetime PD under period-independence approximation: 1 − (1 − PD₁₂m)^T."""
    return 1.0 - (1.0 - pd_12m) ** T


def _d_sicr_threshold(
    ltv_0:   float,
    T:       int,
    beta_0:  float,
    beta_1:  float,
    tau:     float,
    p_flood: float = 1.0,
) -> float:
    """
    Damage ratio at which SICR fires for a single loan (LTV structural model).

    Two framings via p_flood:
      Conditional   (p_flood=1.0): solves PD_LT_stressed / PD_LT_base = tau
      Unconditional (p_flood<1.0): solves PD_adj_LT / PD_LT_base = tau
        where effective threshold on conditional PD = (tau-(1-p_LT))/p_LT

    Returns NaN if unreachable within d ∈ [0, 1).
    """
    pd_12m_base  = _logistic_pd(np.array([ltv_0]), beta_0, beta_1)[0]
    pd_lt_base   = _lifetime_pd(np.array([pd_12m_base]), np.array([T]))[0]

    if p_flood == 1.0:
        target_pd_lt = pd_lt_base * tau
    else:
        p_lt          = 1.0 - (1.0 - p_flood) ** T
        effective_tau = (tau - (1.0 - p_lt)) / p_lt
        target_pd_lt  = pd_lt_base * effective_tau

    if target_pd_lt >= 1.0:
        return np.nan

    def _objective(d: float) -> float:
        ltv_d    = ltv_0 / (1.0 - d)
        pd_12m_d = _logistic_pd(np.array([ltv_d]), beta_0, beta_1)[0]
        pd_lt_d  = _lifetime_pd(np.array([pd_12m_d]), np.array([T]))[0]
        return pd_lt_d - target_pd_lt

    try:
        if _objective(0.999) < 0.0:
            return np.nan
        d_star = optimize.brentq(_objective, 0.0, 0.999, xtol=1e-6)
        return float(d_star)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Main portfolio builder
# ---------------------------------------------------------------------------

def build_portfolio(
    config: dict,
    df_properties: pd.DataFrame,
    fit_results: pd.DataFrame,
    random_seed: int = None,
) -> pd.DataFrame:
    """
    Construct a stylised loan portfolio for the LTV-based structural model.

    One loan is created per property in df_properties. The primary state
    variables (LTV₀, PV₀, maturity) are drawn from distributions defined in
    config['portfolio']. All credit risk quantities are then derived
    structurally — EAD, baseline PD, and baseline LGD are outputs, not inputs.

    Reduced-form characteristics (independent EAD/PD/LGD draws) are appended
    as *_rf columns for use in the sensitivity robustness check (Notebook 05).

    Parameters
    ----------
    config : dict
        Full config loaded from parameters.yaml.
    df_properties : pd.DataFrame
        Clean property data (cfrf_garp_clean.csv). Must contain
        property_rank, minimum_dr, mean_dr, maximum_dr.
    fit_results : pd.DataFrame
        Output of fit_all_distributions() for the triangular primary
        distribution. Must align row-for-row with df_properties.
    random_seed : int, optional
        Overrides config['portfolio']['random_seed'] if provided.

    Returns
    -------
    pd.DataFrame
        One row per loan. Columns:

        Physical risk (from CFRF/GARP data):
          property_rank, d_min, d_mean, d_max, vendor_spread,
          is_zero_damage, tri_mode, tri_clamped

        LTV-structural model — primary draws:
          ltv_0         LTV at origination (Uniform 0.60–0.90)
          pv_0_m        Property value at origination (£m, Lognormal)
          maturity_years  Loan term (years, discrete Uniform 5–25)

        LTV-structural model — derived credit risk:
          ead_m           EAD = LTV₀ × PV₀ (£m)
          baseline_pd_12m  PD₁₂m = logistic(β₀ + β₁ × LTV₀)
          baseline_lgd     LGD at d=0: max(1 − (1−ω)/LTV₀, 0)
          baseline_pd_lt   PD_LT = 1 − (1 − PD₁₂m)ᵀ
          baseline_ecl_m    ECL = PD₁₂m × LGD × EAD (£m, Stage 1 baseline)
          d_sicr_threshold  Conditional d* (p=1.0); NaN if unreachable
          d_sicr_uncond     Unconditional d* (p=annual_probability); always ≥ d_sicr_threshold

        Reduced-form model — sensitivity columns:
          ead_rf_m, baseline_pd_rf, baseline_lgd_rf,
          baseline_pd_lt_rf, baseline_ecl_rf_m
    """
    cfg_p    = config["portfolio"]
    cfg_cr   = config["decision1_credit_risk"]
    cfg_ltv  = cfg_cr["ltv_structural"]
    tau      = cfg_cr["sicr_threshold"]
    p_flood  = cfg_cr["flood_hazard"]["annual_probability"]

    seed = random_seed if random_seed is not None else cfg_p["random_seed"]
    rng  = np.random.default_rng(seed)
    n    = len(df_properties)

    # ── LTV-structural model: primary draws ───────────────────────────────

    # LTV₀ ~ Uniform(low, high)
    ltv_cfg = cfg_p["ltv_origination"]
    ltv_0 = rng.uniform(ltv_cfg["params"]["low"], ltv_cfg["params"]["high"], size=n)

    # PV₀ ~ Lognormal(μ, σ) in £m
    # scipy lognorm parameterised as lognorm(s=σ, scale=exp(μ))
    pv_cfg = cfg_p["property_value"]
    pv_0_m = stats.lognorm.rvs(
        s=pv_cfg["params"]["sigma"],
        scale=np.exp(pv_cfg["params"]["mu"]),
        size=n,
        random_state=rng.integers(2**31),
    )

    # Maturity ~ discrete Uniform(low, high) years
    mat_cfg = cfg_p["maturity"]
    maturity_years = rng.integers(
        mat_cfg["params"]["low"], mat_cfg["params"]["high"] + 1, size=n
    )

    # ── LTV-structural model: derived credit risk quantities ──────────────

    beta_0 = cfg_ltv["beta_0"]
    beta_1 = cfg_ltv["beta_1"]
    omega  = cfg_ltv["omega"]

    # EAD derived structurally
    ead_m = ltv_0 * pv_0_m

    # Baseline PD_12m from logistic at LTV₀ (no climate stress, d=0)
    baseline_pd_12m = _logistic_pd(ltv_0, beta_0, beta_1)

    # Baseline LGD from collateral shortfall formula at d=0
    # Note: loans with LTV₀ < (1−ω) = 0.75 have zero baseline LGD —
    # the collateral fully covers the loan even after workout costs.
    baseline_lgd = _collateral_shortfall_lgd(ltv_0, d=0.0, omega=omega)

    # Baseline lifetime PD (independence approximation)
    baseline_pd_lt = _lifetime_pd(baseline_pd_12m, maturity_years)

    # Baseline Stage 1 ECL (pre-climate)
    baseline_ecl_m = baseline_pd_12m * baseline_lgd * ead_m

    # SICR damage threshold per loan — two framings
    # Conditional  (p=1.0): flood certain; isolates vendor disagreement effect
    # Unconditional (p=annual_probability): flood probabilistic; expected ECL uplift
    d_sicr_cond = np.array([
        _d_sicr_threshold(ltv_0[i], int(maturity_years[i]), beta_0, beta_1, tau,
                          p_flood=1.0)
        for i in range(n)
    ])
    d_sicr_uncond = np.array([
        _d_sicr_threshold(ltv_0[i], int(maturity_years[i]), beta_0, beta_1, tau,
                          p_flood=p_flood)
        for i in range(n)
    ])

    # ── Reduced-form model: independent draws ─────────────────────────────

    ead_rf_cfg = cfg_p["ead"]
    ead_rf_m = rng.uniform(
        ead_rf_cfg["params"]["low"], ead_rf_cfg["params"]["high"], size=n
    )

    pd_rf_cfg = cfg_p["baseline_pd_12m"]
    baseline_pd_rf = stats.lognorm.rvs(
        s=pd_rf_cfg["params"]["sigma"],
        scale=np.exp(pd_rf_cfg["params"]["mu"]),
        size=n,
        random_state=rng.integers(2**31),
    )
    baseline_pd_rf = np.clip(baseline_pd_rf, 1e-6, 0.9999)

    lgd_rf_cfg = cfg_p["baseline_lgd"]
    baseline_lgd_rf = rng.uniform(
        lgd_rf_cfg["params"]["low"], lgd_rf_cfg["params"]["high"], size=n
    )

    baseline_pd_lt_rf  = _lifetime_pd(baseline_pd_rf, maturity_years)
    baseline_ecl_rf_m  = baseline_pd_rf * baseline_lgd_rf * ead_rf_m

    # ── Assemble portfolio DataFrame ──────────────────────────────────────

    portfolio = pd.DataFrame({
        # Physical risk
        "property_rank":     df_properties["property_rank"].values,
        "d_min":             df_properties["minimum_dr"].values,
        "d_mean":            df_properties["mean_dr"].values,
        "d_max":             df_properties["maximum_dr"].values,
        "vendor_spread":     (df_properties["maximum_dr"] - df_properties["minimum_dr"]).values,
        "is_zero_damage":    fit_results["is_zero_damage"].values,
        "tri_mode":          fit_results["tri_mode"].values,
        "tri_clamped":       fit_results["tri_clamped"].values,
        # LTV-structural: primary draws
        "ltv_0":             ltv_0,
        "pv_0_m":            pv_0_m,
        "maturity_years":    maturity_years.astype(int),
        # LTV-structural: derived
        "ead_m":             ead_m,
        "baseline_pd_12m":   baseline_pd_12m,
        "baseline_lgd":      baseline_lgd,
        "baseline_pd_lt":    baseline_pd_lt,
        "baseline_ecl_m":    baseline_ecl_m,
        "d_sicr_threshold":  d_sicr_cond,    # conditional (p=1.0)
        "d_sicr_uncond":     d_sicr_uncond,  # unconditional (p=annual_probability)
        # Reduced-form: sensitivity columns
        "ead_rf_m":          ead_rf_m,
        "baseline_pd_rf":    baseline_pd_rf,
        "baseline_lgd_rf":   baseline_lgd_rf,
        "baseline_pd_lt_rf": baseline_pd_lt_rf,
        "baseline_ecl_rf_m": baseline_ecl_rf_m,
    })

    return portfolio
