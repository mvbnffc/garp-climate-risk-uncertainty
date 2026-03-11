"""
Distribution fitting for vendor uncertainty in physical climate risk estimates.

Fits Beta, Triangular, and Uniform distributions to (min, mean, max) summary
statistics from the CFRF/GARP benchmarking study (13 vendors, 100 properties).

Key design choices:
- Damage ratios are bounded on [0, 1]
- We only observe min, mean, max across vendors — not the full distribution
- Multiple quantile interpretations are supported for the min/max
- Properties with zero damage across all vendors are flagged, not fitted
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from typing import Dict, Tuple, Optional, Literal
from dataclasses import dataclass


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FittedDistribution:
    """Container for a fitted distribution and its diagnostics."""
    family: str                     # "beta", "triangular", "uniform"
    params: Dict                    # family-specific parameters
    scipy_dist: object              # frozen scipy.stats distribution for sampling
    mean: float                     # analytical mean of fitted distribution
    variance: float                 # analytical variance
    skewness: float                 # analytical skewness
    quantile_05: float              # 5th percentile
    quantile_95: float              # 95th percentile
    fit_success: bool               # whether fitting converged / was valid
    fit_notes: str                  # any warnings or issues


# ──────────────────────────────────────────────────────────────────────────────
# Beta distribution fitting
# ──────────────────────────────────────────────────────────────────────────────

def fit_beta_quantile_matching(
    d_min: float,
    d_mean: float,
    d_max: float,
    lower_q: float = 0.0538,
    upper_q: float = 0.9462,
    interpretation: Literal["order_statistic", "quantile_approx", "extremes"] = "order_statistic",
) -> FittedDistribution:
    """
    Fit a Beta distribution to (min, mean, max) damage ratio summary statistics.

    For 'order_statistic' and 'quantile_approx' interpretations:
        Minimise squared error between the Beta quantiles at (lower_q, upper_q)
        and the observed (min, max), subject to matching the mean.

    For 'extremes' interpretation:
        Fit a scaled Beta on [min, max] with the correct mean. The Beta is
        defined on [0,1] and then linearly mapped to [min, max].

    Parameters
    ----------
    d_min, d_mean, d_max : float
        Observed minimum, mean, and maximum damage ratio across vendors.
    lower_q, upper_q : float
        Quantile levels to which min and max correspond.
    interpretation : str
        How to treat min/max. See config/parameters.yaml for details.

    Returns
    -------
    FittedDistribution
    """
    notes = []

    # ── Edge cases ──
    if d_max == 0 and d_mean == 0 and d_min == 0:
        return FittedDistribution(
            family="beta", params={"a": np.nan, "b": np.nan},
            scipy_dist=None, mean=0.0, variance=0.0, skewness=0.0,
            quantile_05=0.0, quantile_95=0.0,
            fit_success=False, fit_notes="Zero damage across all vendors"
        )

    if d_mean <= 0:
        # Some properties have mean=0 but max>0; can't fit a proper Beta
        notes.append("Mean is zero but max > 0; degenerate case")
        return FittedDistribution(
            family="beta", params={"a": np.nan, "b": np.nan},
            scipy_dist=None, mean=0.0, variance=0.0, skewness=0.0,
            quantile_05=0.0, quantile_95=d_max,
            fit_success=False, fit_notes="; ".join(notes)
        )

    if d_min == d_max:
        # All vendors agree — point mass
        notes.append("No vendor dispersion (min == max)")
        return FittedDistribution(
            family="beta", params={"a": np.nan, "b": np.nan},
            scipy_dist=None, mean=d_mean, variance=0.0, skewness=0.0,
            quantile_05=d_mean, quantile_95=d_mean,
            fit_success=False, fit_notes="; ".join(notes)
        )

    # ── Extremes interpretation: scaled Beta on [d_min, d_max] ──
    if interpretation == "extremes":
        return _fit_beta_scaled(d_min, d_mean, d_max)

    # ── Quantile-matching interpretation ──
    # We solve for (a, b) such that:
    #   Beta(a,b).ppf(lower_q) ≈ d_min
    #   Beta(a,b).ppf(upper_q) ≈ d_max
    #   Beta(a,b).mean() = a/(a+b) ≈ d_mean
    #
    # Three constraints, two parameters → least-squares with mean as hard constraint
    # Parameterise as (a, b) with a = d_mean * nu, b = (1 - d_mean) * nu
    # where nu = a + b is the "concentration" parameter. This enforces the mean exactly.
    # Then optimise nu to match the quantiles.

    # Clamp mean to (0, 1) open interval
    mu = np.clip(d_mean, 1e-6, 1 - 1e-6)
    # Clamp targets
    target_lo = np.clip(d_min, 0.0, mu - 1e-6)
    target_hi = np.clip(d_max, mu + 1e-6, 1.0)

    def objective(log_nu):
        nu = np.exp(log_nu)
        a = mu * nu
        b = (1 - mu) * nu
        if a <= 0 or b <= 0:
            return 1e10
        try:
            q_lo = stats.beta.ppf(lower_q, a, b)
            q_hi = stats.beta.ppf(upper_q, a, b)
            # Weighted least squares on quantile match
            err = ((q_lo - target_lo) ** 2 + (q_hi - target_hi) ** 2)
            return err
        except Exception:
            return 1e10

    # Search over a range of concentration parameters
    # Low nu → wide distribution; high nu → tight around mean
    result = optimize.minimize_scalar(objective, bounds=(-2, 10), method="bounded")

    if not result.success and result.fun > 0.01:
        notes.append(f"Optimisation did not converge well (residual={result.fun:.4f})")

    nu = np.exp(result.x)
    a = mu * nu
    b = (1 - mu) * nu

    # Validate
    if a < 1e-4 or b < 1e-4:
        notes.append(f"Extreme shape parameters (a={a:.4f}, b={b:.4f})")

    frozen = stats.beta(a, b)

    return FittedDistribution(
        family="beta",
        params={"a": a, "b": b, "nu": nu},
        scipy_dist=frozen,
        mean=frozen.mean(),
        variance=frozen.var(),
        skewness=frozen.stats(moments="s").item() if hasattr(frozen.stats(moments="s"), 'item') else float(frozen.stats(moments="s")),
        quantile_05=frozen.ppf(0.05),
        quantile_95=frozen.ppf(0.95),
        fit_success=True,
        fit_notes="; ".join(notes) if notes else "OK"
    )


def _fit_beta_scaled(d_min: float, d_mean: float, d_max: float) -> FittedDistribution:
    """
    Fit a Beta distribution on [d_min, d_max] by rescaling to [0, 1].

    The rescaled mean is mu_s = (d_mean - d_min) / (d_max - d_min).
    We then fit a Beta(a, b) on [0, 1] with mean = mu_s.
    Since we have only one free constraint (mean), we use a heuristic:
    set nu = a + b = 4 (moderate concentration) as a default, then
    a = mu_s * nu, b = (1 - mu_s) * nu.

    For sampling, the scaled distribution is: X = d_min + (d_max - d_min) * Beta(a, b)
    """
    notes = []
    span = d_max - d_min
    if span <= 0:
        return FittedDistribution(
            family="beta", params={}, scipy_dist=None,
            mean=d_mean, variance=0.0, skewness=0.0,
            quantile_05=d_mean, quantile_95=d_mean,
            fit_success=False, fit_notes="Zero range for extremes interpretation"
        )

    mu_s = (d_mean - d_min) / span
    mu_s = np.clip(mu_s, 1e-4, 1 - 1e-4)

    # With only the mean as a constraint on [0,1], we need an assumption about spread.
    # Use nu = 2 + k where k reflects that 13 vendors give moderate information.
    # nu = 4 is a reasonable default (mild concentration around mean).
    # Lower nu → flatter; higher nu → more peaked.
    # We provide nu as a parameter that can be varied in sensitivity analysis.
    nu = 4.0
    a = mu_s * nu
    b = (1 - mu_s) * nu

    # The scipy distribution for sampling: use loc and scale
    frozen = stats.beta(a, b, loc=d_min, scale=span)

    return FittedDistribution(
        family="beta",
        params={"a": a, "b": b, "nu": nu, "loc": d_min, "scale": span},
        scipy_dist=frozen,
        mean=frozen.mean(),
        variance=frozen.var(),
        skewness=float(frozen.stats(moments="s")),
        quantile_05=frozen.ppf(0.05),
        quantile_95=frozen.ppf(0.95),
        fit_success=True,
        fit_notes="; ".join(notes) if notes else "OK (extremes interpretation, nu=4)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Triangular distribution fitting
# ──────────────────────────────────────────────────────────────────────────────

def fit_triangular_from_moments(
    d_min: float,
    d_mean: float,
    d_max: float,
) -> FittedDistribution:
    """
    Fit a Triangular distribution from (min, mean, max).

    For a triangular distribution on [a, b] with mode c:
        mean = (a + b + c) / 3
    Therefore:
        mode = 3 * mean - min - max

    If mode falls outside [min, max], the triangular is inconsistent with the
    data. In that case, we clamp to the nearest boundary and flag.

    scipy.stats.triang parameterisation:
        c_param = (mode - min) / (max - min)   # shape parameter in [0, 1]
        loc = min
        scale = max - min
    """
    notes = []

    # Edge cases
    if d_max == 0 and d_mean == 0:
        return FittedDistribution(
            family="triangular", params={}, scipy_dist=None,
            mean=0.0, variance=0.0, skewness=0.0,
            quantile_05=0.0, quantile_95=0.0,
            fit_success=False, fit_notes="Zero damage across all vendors"
        )

    if d_min == d_max:
        return FittedDistribution(
            family="triangular", params={}, scipy_dist=None,
            mean=d_mean, variance=0.0, skewness=0.0,
            quantile_05=d_mean, quantile_95=d_mean,
            fit_success=False, fit_notes="No vendor dispersion (min == max)"
        )

    mode = 3 * d_mean - d_min - d_max
    clamped = False

    if mode < d_min:
        notes.append(f"Implied mode ({mode:.4f}) < min ({d_min:.4f}); clamped to min")
        mode = d_min
        clamped = True
    elif mode > d_max:
        notes.append(f"Implied mode ({mode:.4f}) > max ({d_max:.4f}); clamped to max")
        mode = d_max
        clamped = True

    span = d_max - d_min
    c_param = (mode - d_min) / span  # scipy shape parameter

    frozen = stats.triang(c_param, loc=d_min, scale=span)

    return FittedDistribution(
        family="triangular",
        params={"min": d_min, "mode": mode, "max": d_max, "c_scipy": c_param, "clamped": clamped},
        scipy_dist=frozen,
        mean=frozen.mean(),
        variance=frozen.var(),
        skewness=float(frozen.stats(moments="s")),
        quantile_05=frozen.ppf(0.05),
        quantile_95=frozen.ppf(0.95),
        fit_success=True,
        fit_notes="; ".join(notes) if notes else "OK"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Uniform distribution
# ──────────────────────────────────────────────────────────────────────────────

def fit_uniform(
    d_min: float,
    d_mean: float,
    d_max: float,
) -> FittedDistribution:
    """
    Uniform distribution on [d_min, d_max].

    This is the maximum-entropy distribution given only the support bounds,
    and represents the most dispersed (most uncertain) interpretation of
    vendor disagreement. The mean of Uniform(a, b) = (a + b) / 2, which
    will generally NOT match the observed mean. This mismatch is informative:
    it shows the degree to which the vendor distribution is asymmetric.

    We report the implied mean alongside the observed mean for comparison.
    """
    notes = []

    if d_max == 0 and d_mean == 0:
        return FittedDistribution(
            family="uniform", params={}, scipy_dist=None,
            mean=0.0, variance=0.0, skewness=0.0,
            quantile_05=0.0, quantile_95=0.0,
            fit_success=False, fit_notes="Zero damage across all vendors"
        )

    if d_min == d_max:
        return FittedDistribution(
            family="uniform", params={}, scipy_dist=None,
            mean=d_mean, variance=0.0, skewness=0.0,
            quantile_05=d_mean, quantile_95=d_mean,
            fit_success=False, fit_notes="No vendor dispersion"
        )

    frozen = stats.uniform(loc=d_min, scale=d_max - d_min)
    implied_mean = frozen.mean()
    mean_mismatch = implied_mean - d_mean

    if abs(mean_mismatch) > 0.05:
        notes.append(
            f"Uniform mean ({implied_mean:.3f}) differs from observed mean "
            f"({d_mean:.3f}) by {mean_mismatch:+.3f}"
        )

    return FittedDistribution(
        family="uniform",
        params={"min": d_min, "max": d_max, "mean_mismatch": mean_mismatch},
        scipy_dist=frozen,
        mean=implied_mean,
        variance=frozen.var(),
        skewness=0.0,  # uniform is symmetric
        quantile_05=frozen.ppf(0.05),
        quantile_95=frozen.ppf(0.95),
        fit_success=True,
        fit_notes="; ".join(notes) if notes else "OK"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Batch fitting
# ──────────────────────────────────────────────────────────────────────────────

def fit_all_distributions(
    df: pd.DataFrame,
    min_col: str = "minimum_dr",
    mean_col: str = "mean_dr",
    max_col: str = "maximum_dr",
    beta_lower_q: float = 0.0538,
    beta_upper_q: float = 0.9462,
    beta_interpretation: str = "order_statistic",
) -> pd.DataFrame:
    """
    Fit Beta, Triangular, and Uniform distributions to all properties.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns for min, mean, max damage ratios.
    beta_lower_q, beta_upper_q : float
        Quantile levels for Beta fitting.
    beta_interpretation : str
        One of "order_statistic", "quantile_approx", "extremes".

    Returns
    -------
    pd.DataFrame
        One row per property with fitted parameters, diagnostics, and flags.
    """
    records = []

    for _, row in df.iterrows():
        d_min = row[min_col]
        d_mean = row[mean_col]
        d_max = row[max_col]
        prop_rank = row.get("property_rank", None)

        # Flag zero-damage properties
        is_zero = (d_max == 0) and (d_mean == 0) and (d_min == 0)

        # Fit each distribution
        beta_fit = fit_beta_quantile_matching(
            d_min, d_mean, d_max,
            lower_q=beta_lower_q, upper_q=beta_upper_q,
            interpretation=beta_interpretation,
        )
        tri_fit = fit_triangular_from_moments(d_min, d_mean, d_max)
        uni_fit = fit_uniform(d_min, d_mean, d_max)

        record = {
            "property_rank": prop_rank,
            "d_min": d_min,
            "d_mean": d_mean,
            "d_max": d_max,
            "is_zero_damage": is_zero,
            # Beta
            "beta_a": beta_fit.params.get("a", np.nan),
            "beta_b": beta_fit.params.get("b", np.nan),
            "beta_nu": beta_fit.params.get("nu", np.nan),
            "beta_mean": beta_fit.mean,
            "beta_var": beta_fit.variance,
            "beta_skew": beta_fit.skewness,
            "beta_q05": beta_fit.quantile_05,
            "beta_q95": beta_fit.quantile_95,
            "beta_success": beta_fit.fit_success,
            "beta_notes": beta_fit.fit_notes,
            # Triangular
            "tri_mode": tri_fit.params.get("mode", np.nan),
            "tri_clamped": tri_fit.params.get("clamped", False),
            "tri_mean": tri_fit.mean,
            "tri_var": tri_fit.variance,
            "tri_skew": tri_fit.skewness,
            "tri_q05": tri_fit.quantile_05,
            "tri_q95": tri_fit.quantile_95,
            "tri_success": tri_fit.fit_success,
            "tri_notes": tri_fit.fit_notes,
            # Uniform
            "uni_mean": uni_fit.mean,
            "uni_var": uni_fit.variance,
            "uni_mean_mismatch": uni_fit.params.get("mean_mismatch", np.nan),
            "uni_q05": uni_fit.quantile_05,
            "uni_q95": uni_fit.quantile_95,
            "uni_success": uni_fit.fit_success,
            "uni_notes": uni_fit.fit_notes,
        }
        records.append(record)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────────

def sample_vendor_uncertainty(
    df: pd.DataFrame,
    fit_results: pd.DataFrame,
    distribution: Literal["beta", "triangular", "uniform"] = "beta",
    n_samples: int = 10000,
    correlation: Literal["independent", "perfectly_correlated"] = "independent",
    min_col: str = "minimum_dr",
    mean_col: str = "mean_dr",
    max_col: str = "maximum_dr",
    beta_lower_q: float = 0.0538,
    beta_upper_q: float = 0.9462,
    beta_interpretation: str = "order_statistic",
    random_state: int = 42,
) -> np.ndarray:
    """
    Draw Monte Carlo samples of damage ratios for all properties.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with min/mean/max columns.
    fit_results : pd.DataFrame
        Output of fit_all_distributions().
    distribution : str
        Which fitted distribution to sample from.
    n_samples : int
        Number of Monte Carlo draws.
    correlation : str
        "independent" — independent draws per property.
        "perfectly_correlated" — single quantile mapped through all marginals.
    random_state : int
        RNG seed.

    Returns
    -------
    np.ndarray of shape (n_samples, n_properties)
        Sampled damage ratios.
    """
    rng = np.random.default_rng(random_state)
    n_props = len(df)
    samples = np.zeros((n_samples, n_props))

    # Build frozen distributions for each property
    frozen_dists = []
    for idx, (_, row) in enumerate(df.iterrows()):
        d_min = row[min_col]
        d_mean = row[mean_col]
        d_max = row[max_col]

        if fit_results.iloc[idx]["is_zero_damage"]:
            frozen_dists.append(None)  # point mass at 0
            continue

        if distribution == "beta":
            fd = fit_beta_quantile_matching(
                d_min, d_mean, d_max,
                lower_q=beta_lower_q, upper_q=beta_upper_q,
                interpretation=beta_interpretation,
            )
        elif distribution == "triangular":
            fd = fit_triangular_from_moments(d_min, d_mean, d_max)
        elif distribution == "uniform":
            fd = fit_uniform(d_min, d_mean, d_max)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        frozen_dists.append(fd.scipy_dist if fd.fit_success else None)

    # Sample
    if correlation == "independent":
        for j, dist in enumerate(frozen_dists):
            if dist is None:
                samples[:, j] = 0.0
            else:
                samples[:, j] = dist.rvs(size=n_samples, random_state=rng)
    elif correlation == "perfectly_correlated":
        # Draw a single uniform quantile per simulation
        u = rng.uniform(0, 1, size=n_samples)
        for j, dist in enumerate(frozen_dists):
            if dist is None:
                samples[:, j] = 0.0
            else:
                samples[:, j] = dist.ppf(u)
    else:
        raise ValueError(f"Unknown correlation: {correlation}")

    # Clip to [0, 1] for safety
    samples = np.clip(samples, 0.0, 1.0)

    return samples
