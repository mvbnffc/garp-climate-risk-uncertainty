"""
Credit risk transmission — LTV structural model.

Maps a damage ratio d ∈ [0, 1) through to PD, LGD, and ECL for a single loan
or an array of loans. Two models are provided:

  Primary   — LTV structural: damage → LTV → logistic PD, collateral shortfall LGD
  Sensitivity — Reduced form:  damage → PD_0 * exp(α*d),  LGD_0 + λ*d

All functions accept numpy arrays and are designed to vectorise cleanly across
both the damage dimension (for curve plotting) and the loan dimension (for
portfolio simulation).
"""

import numpy as np
from scipy import optimize


# ── LTV structural model ────────────────────────────────────────────────────

def logistic_pd(ltv: np.ndarray, beta_0: float, beta_1: float) -> np.ndarray:
    """
    12-month PD as a logistic function of current LTV.

    PD_12m = 1 / (1 + exp(-(β₀ + β₁ × LTV)))

    Parameters
    ----------
    ltv : array-like
        Current loan-to-value ratio (stressed or baseline).
    beta_0, beta_1 : float
        Logistic intercept and slope from config.

    Returns
    -------
    np.ndarray of PD values in [0, 1].
    """
    ltv = np.asarray(ltv, dtype=float)
    return 1.0 / (1.0 + np.exp(-(beta_0 + beta_1 * ltv)))


def collateral_shortfall_lgd(
    ltv_0: np.ndarray, d: np.ndarray, omega: float
) -> np.ndarray:
    """
    LGD from collateral shortfall net of workout costs.

    LGD(d) = max(1 - (1-d)(1-ω) / LTV₀, 0)

    The damaged property is sold at a discount ω (foreclosure costs, fire-sale).
    The loss is whatever the net proceeds fail to cover of the outstanding balance.

    Parameters
    ----------
    ltv_0 : array-like
        LTV at origination (one value per loan).
    d : array-like
        Damage ratio (broadcast against ltv_0).
    omega : float
        Workout / fire-sale discount from config.

    Returns
    -------
    np.ndarray of LGD values in [0, 1].
    """
    ltv_0 = np.asarray(ltv_0, dtype=float)
    d     = np.asarray(d,     dtype=float)
    return np.maximum(1.0 - (1.0 - d) * (1.0 - omega) / ltv_0, 0.0)


def lifetime_pd(pd_12m: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Lifetime PD under period-independence approximation.

    PD_LT = 1 - (1 - PD_12m)^T

    Parameters
    ----------
    pd_12m : array-like
        Annual default probability.
    T : array-like
        Loan maturity in years (broadcast against pd_12m).

    Returns
    -------
    np.ndarray of lifetime PD values in [0, 1].
    """
    pd_12m = np.asarray(pd_12m, dtype=float)
    T      = np.asarray(T,      dtype=float)
    return 1.0 - (1.0 - pd_12m) ** T


def stressed_ltv(ltv_0: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Current LTV after damage: LTV(d) = LTV₀ / (1 - d).

    Undefined at d=1 (total destruction). Callers should restrict d < 1.
    """
    ltv_0 = np.asarray(ltv_0, dtype=float)
    d     = np.asarray(d,     dtype=float)
    return ltv_0 / (1.0 - d)


def compute_stressed_metrics(
    d:      np.ndarray,
    ltv_0:  np.ndarray,
    T:      np.ndarray,
    beta_0: float,
    beta_1: float,
    omega:  float,
) -> dict:
    """
    Full LTV structural transmission for given damage ratios and loan parameters.

    Broadcasts over (d, ltv_0, T) — shapes must be compatible.

    Parameters
    ----------
    d : array-like
        Damage ratio(s).
    ltv_0 : array-like
        LTV at origination (per loan).
    T : array-like
        Loan maturity in years (per loan).
    beta_0, beta_1 : float
        Logistic PD parameters.
    omega : float
        Workout discount.

    Returns
    -------
    dict with keys:
        ltv_d      current LTV after damage
        pd_12m     stressed 12-month PD
        lgd        stressed LGD
        pd_lt      stressed lifetime PD
    """
    d     = np.asarray(d,     dtype=float)
    ltv_0 = np.asarray(ltv_0, dtype=float)
    T     = np.asarray(T,     dtype=float)

    ltv_d  = stressed_ltv(ltv_0, d)
    pd_12m = logistic_pd(ltv_d, beta_0, beta_1)
    lgd    = collateral_shortfall_lgd(ltv_0, d, omega)
    pd_lt  = lifetime_pd(pd_12m, T)

    return {"ltv_d": ltv_d, "pd_12m": pd_12m, "lgd": lgd, "pd_lt": pd_lt}


def compute_ecl(
    d:           np.ndarray,
    ltv_0:       np.ndarray,
    T:           np.ndarray,
    ead:         np.ndarray,
    pd_lt_base:  np.ndarray,
    beta_0:      float,
    beta_1:      float,
    omega:       float,
    tau:         float,
    p_flood:     float = 1.0,
) -> dict:
    """
    Stage-dependent ECL under the LTV structural model.

    Two probability framings are supported via p_flood:

      Conditional   (p_flood=1.0, default):
        Assumes the flood definitely occurs. PD used in ECL is the stressed
        PD directly. SICR ratio = PD_LT_stressed / PD_LT_baseline.
        Isolates the vendor disagreement effect without probability dilution.

      Unconditional (p_flood = annual exceedance probability, e.g. 0.005):
        Weights stressed and baseline PDs by flood probability.
        PD_adj_12m = (1 − p) × PD_base_12m  + p × PD_stressed_12m
        PD_adj_LT  = (1 − p_LT) × PD_base_LT + p_LT × PD_stressed_LT
        where p_LT = 1 − (1 − p_flood)^T  (varies per loan by maturity).
        SICR ratio = PD_adj_LT / PD_LT_baseline.
        Gives the expected ECL uplift a bank would actually book.

    Parameters
    ----------
    d : array-like
        Damage ratio(s).
    ltv_0 : array-like
        LTV at origination (per loan).
    T : array-like
        Maturity in years (per loan).
    ead : array-like
        Exposure at default in £m (per loan).
    pd_lt_base : array-like
        Baseline lifetime PD at d=0 (per loan) — SICR test denominator.
    beta_0, beta_1, omega, tau : float
        Model parameters from config.
    p_flood : float, optional
        Annual flood exceedance probability. Default 1.0 (conditional).

    Returns
    -------
    dict with keys:
        pd_12m_ecl   PD used in ECL calculation (adjusted or stressed)
        pd_lt_ecl    Lifetime PD used in ECL calculation
        lgd          Stressed LGD
        stage        1 or 2 (integer array)
        ecl_m        ECL in £m
        sicr_ratio   PD_adj_LT / PD_LT_baseline
    """
    metrics = compute_stressed_metrics(d, ltv_0, T, beta_0, beta_1, omega)
    pd_12m_stressed = metrics["pd_12m"]
    lgd             = metrics["lgd"]
    pd_lt_stressed  = metrics["pd_lt"]

    pd_lt_base = np.asarray(pd_lt_base, dtype=float)
    ead        = np.asarray(ead,        dtype=float)
    T_arr      = np.asarray(T,          dtype=float)

    if p_flood == 1.0:
        # Conditional: use stressed PDs directly
        pd_12m_ecl = pd_12m_stressed
        pd_lt_ecl  = pd_lt_stressed
    else:
        # Unconditional: probability-weight stressed and baseline PDs
        p_lt       = 1.0 - (1.0 - p_flood) ** T_arr
        pd_12m_base = logistic_pd(np.asarray(ltv_0, dtype=float), beta_0, beta_1)
        pd_12m_ecl = (1.0 - p_flood) * pd_12m_base + p_flood * pd_12m_stressed
        pd_lt_ecl  = (1.0 - p_lt)    * pd_lt_base  + p_lt    * pd_lt_stressed

    sicr_ratio = np.where(pd_lt_base > 0, pd_lt_ecl / pd_lt_base, 1.0)
    in_stage2  = sicr_ratio > tau

    ecl_stage1 = pd_12m_ecl * lgd * ead
    ecl_stage2 = pd_lt_ecl  * lgd * ead
    ecl_m      = np.where(in_stage2, ecl_stage2, ecl_stage1)

    return {
        "pd_12m_ecl":  pd_12m_ecl,
        "pd_lt_ecl":   pd_lt_ecl,
        "lgd":         lgd,
        "stage":       np.where(in_stage2, 2, 1).astype(int),
        "ecl_m":       ecl_m,
        "sicr_ratio":  sicr_ratio,
    }


def sicr_damage_threshold(
    ltv_0:   float,
    T:       int,
    beta_0:  float,
    beta_1:  float,
    tau:     float,
    p_flood: float = 1.0,
) -> float:
    """
    Damage ratio d* at which SICR fires for a single loan.

    Two framings via p_flood (matching compute_ecl):

      Conditional (p_flood=1.0):
        Solves: PD_LT_stressed(d*) / PD_LT_baseline = tau
        The stressed conditional PD must multiply by tau relative to baseline.

      Unconditional (p_flood = annual probability, e.g. 0.005):
        Solves: PD_adj_LT(d*) / PD_LT_baseline = tau
        where PD_adj_LT = (1-p_LT)*PD_base_LT + p_LT*PD_stressed_LT
        Rearranges to: PD_stressed_LT > effective_tau * PD_base_LT
        where effective_tau = (tau - (1-p_LT)) / p_LT  (always > tau)
        This is a much higher bar — e.g. tau=2, T=20y → effective_tau ≈ 11.5

    Returns NaN if the threshold cannot be reached within d ∈ [0, 1).

    Parameters
    ----------
    ltv_0 : float
        LTV at origination.
    T : int
        Loan maturity in years.
    beta_0, beta_1 : float
        Logistic parameters.
    tau : float
        SICR multiplier threshold (e.g. 2.0 for EBA doubling rule).
    p_flood : float, optional
        Annual flood exceedance probability. Default 1.0 (conditional).
    """
    pd_12m_base = logistic_pd(np.array([ltv_0]), beta_0, beta_1)[0]
    pd_lt_base  = lifetime_pd(np.array([pd_12m_base]), np.array([T]))[0]

    if p_flood == 1.0:
        # Conditional: PD_LT_stressed must reach tau × PD_LT_baseline
        target = pd_lt_base * tau
    else:
        # Unconditional: rearrange PD_adj_LT / PD_LT_base = tau
        # → PD_stressed_LT = (tau - (1-p_LT)) / p_LT × PD_LT_base
        p_lt           = 1.0 - (1.0 - p_flood) ** T
        effective_tau  = (tau - (1.0 - p_lt)) / p_lt
        target         = pd_lt_base * effective_tau

    if target >= 1.0:
        return np.nan

    def _f(d: float) -> float:
        ltv_d    = ltv_0 / (1.0 - d)
        pd_12m_d = logistic_pd(np.array([ltv_d]), beta_0, beta_1)[0]
        pd_lt_d  = lifetime_pd(np.array([pd_12m_d]), np.array([T]))[0]
        return pd_lt_d - target

    try:
        if _f(0.999) < 0.0:
            return np.nan
        return float(optimize.brentq(_f, 0.0, 0.999, xtol=1e-6))
    except ValueError:
        return np.nan


# ── Reduced-form (sensitivity) model ────────────────────────────────────────

def reduced_form_pd(
    pd_0: np.ndarray,
    d: np.ndarray,
    alpha: float,
    functional_form: str = "exponential",
) -> np.ndarray:
    """
    Stressed PD under the reduced-form model. Clipped to 1.0.

    Two functional forms are supported (controlled by ``functional_form``):

    ``"exponential"`` (default):
        PD_12m(d) = PD_0 × exp(α × d)
        Convex and unbounded in α; standard ad-hoc choice in the literature.

    ``"linear"``:
        PD_12m(d) = PD_0 × (1 + α × d)
        Affine scaling; always gives PD_12m(0) = PD_0 and PD_12m(1) = PD_0×(1+α).
        More transparent sensitivity: α is the fractional PD uplift per unit damage.

    Parameters
    ----------
    pd_0 : array-like
        Baseline 12-month PD (per loan).
    d : array-like
        Damage ratio.
    alpha : float
        Sensitivity parameter.
    functional_form : {"exponential", "linear"}
        Which functional form to apply.

    Returns
    -------
    np.ndarray of stressed PD values clipped to [0, 1].
    """
    pd_0 = np.asarray(pd_0, dtype=float)
    d    = np.asarray(d,    dtype=float)

    if functional_form == "exponential":
        stressed = pd_0 * np.exp(alpha * d)
    elif functional_form == "linear":
        stressed = pd_0 * (1.0 + alpha * d)
    else:
        raise ValueError(
            f"Unknown functional_form '{functional_form}'. "
            "Choose 'exponential' or 'linear'."
        )
    return np.minimum(stressed, 1.0)


def reduced_form_lgd(lgd_0: np.ndarray, d: np.ndarray, lambda_lgd: float) -> np.ndarray:
    """
    Stressed LGD under reduced-form model: LGD(d) = min(LGD_0 + λ × d, 1).
    """
    return np.minimum(np.asarray(lgd_0) + lambda_lgd * np.asarray(d), 1.0)
