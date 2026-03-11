"""
Stylised loan portfolio construction.

Draws heterogeneous loan characteristics (EAD, maturity, baseline PD, LGD)
from parametric distributions defined in config/parameters.yaml, one loan per
CFRF/GARP property. The 1:1 mapping preserves the physical risk profile of
each property through to the financial simulation stage.
"""

import numpy as np
import pandas as pd
from scipy import stats


def build_portfolio(
    config: dict,
    df_properties: pd.DataFrame,
    fit_results: pd.DataFrame,
    random_seed: int = None,
) -> pd.DataFrame:
    """
    Construct a stylised loan portfolio.

    One loan is created per property in df_properties, with loan characteristics
    drawn independently from the distributions specified in config['portfolio'].
    Baseline lifetime PD is derived from baseline 12-month PD and maturity.

    Parameters
    ----------
    config : dict
        Full config loaded from parameters.yaml.
    df_properties : pd.DataFrame
        Clean property data (cfrf_garp_clean.csv). Must contain
        property_rank, minimum_dr, mean_dr, maximum_dr.
    fit_results : pd.DataFrame
        Output of fit_all_distributions() — triangular fit diagnostics.
        Must align row-for-row with df_properties.
    random_seed : int, optional
        Overrides config['portfolio']['random_seed'] if provided.

    Returns
    -------
    pd.DataFrame
        One row per loan with columns:
          property_rank, ead_m, maturity_years, baseline_pd_12m,
          baseline_lgd, baseline_pd_lt, baseline_ecl_m,
          d_min, d_mean, d_max, vendor_spread,
          is_zero_damage, tri_mode, tri_clamped
    """
    cfg = config["portfolio"]
    seed = random_seed if random_seed is not None else cfg["random_seed"]
    rng = np.random.default_rng(seed)
    n = len(df_properties)

    # ── Draw loan characteristics ──────────────────────────────────────────

    # EAD: Uniform(low, high) in £m
    ead_cfg = cfg["ead"]
    ead_m = rng.uniform(ead_cfg["params"]["low"], ead_cfg["params"]["high"], size=n)

    # Maturity: discrete Uniform(low, high) in years
    mat_cfg = cfg["maturity"]
    maturity_years = rng.integers(
        mat_cfg["params"]["low"], mat_cfg["params"]["high"] + 1, size=n
    )

    # Baseline 12-month PD: Lognormal
    # scipy lognorm(s=sigma, scale=exp(mu)) where mu, sigma are normal parameters
    pd_cfg = cfg["baseline_pd_12m"]
    mu_pd = pd_cfg["params"]["mu"]
    sigma_pd = pd_cfg["params"]["sigma"]
    baseline_pd_12m = stats.lognorm.rvs(
        s=sigma_pd, scale=np.exp(mu_pd), size=n, random_state=rng.integers(2**31)
    )
    # Clip to (0, 1) — lognormal is positive but can theoretically exceed 1
    baseline_pd_12m = np.clip(baseline_pd_12m, 1e-6, 0.9999)

    # Baseline LGD: Uniform(low, high)
    lgd_cfg = cfg["baseline_lgd"]
    baseline_lgd = rng.uniform(
        lgd_cfg["params"]["low"], lgd_cfg["params"]["high"], size=n
    )

    # ── Derived quantities ─────────────────────────────────────────────────

    # Lifetime PD (independence approximation)
    baseline_pd_lt = 1.0 - (1.0 - baseline_pd_12m) ** maturity_years

    # Baseline ECL (Stage 1, pre-climate): PD_12m × LGD × EAD
    baseline_ecl_m = baseline_pd_12m * baseline_lgd * ead_m

    # ── Physical risk fields ───────────────────────────────────────────────

    portfolio = pd.DataFrame({
        "property_rank":    df_properties["property_rank"].values,
        "ead_m":            ead_m,
        "maturity_years":   maturity_years.astype(int),
        "baseline_pd_12m":  baseline_pd_12m,
        "baseline_lgd":     baseline_lgd,
        "baseline_pd_lt":   baseline_pd_lt,
        "baseline_ecl_m":   baseline_ecl_m,
        "d_min":            df_properties["minimum_dr"].values,
        "d_mean":           df_properties["mean_dr"].values,
        "d_max":            df_properties["maximum_dr"].values,
        "vendor_spread":    (df_properties["maximum_dr"] - df_properties["minimum_dr"]).values,
        "is_zero_damage":   fit_results["is_zero_damage"].values,
        "tri_mode":         fit_results["tri_mode"].values,
        "tri_clamped":      fit_results["tri_clamped"].values,
    })

    return portfolio
