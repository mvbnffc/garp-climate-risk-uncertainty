"""
Secured lending transmission — Basel III SA CRE risk-weight model.

Maps a damage ratio d ∈ [0, 1) through to LTV, lending outcome bucket,
risk-weighted asset (RWA), and capital requirement. No PD/LGD model is used —
the transmission chain is purely through collateral value and current LTV.

Transmission chain:
    d_iv  (vendor damage ratio for property i, vendor v)
      → V_iv  = V_i0 × (1 − θ × d)          collateral value after flood damage
      → LTV_iv = Loan_i / V_iv               current loan-to-value
      → outcome ∈ {0, 1, 2}                  standard / conditional / reject
      → RWA_i = Loan_i × RW(outcome_i)       risk-weighted asset
      → Capital_i = RWA_i × capital_ratio

Two LTV cliff thresholds (LTV̄₁ and LTV̄₂) replace the single SICR cliff in
Decision 1. Vendor disagreement that causes impaired LTV to straddle either
threshold directly flips the regulatory capital requirement.

All functions accept numpy arrays and vectorise cleanly across both the damage
dimension (n_simulations) and the loan dimension (n_properties).
"""

import numpy as np


def impaired_collateral_value(v0: np.ndarray, d: np.ndarray, theta: float) -> np.ndarray:
    """
    Collateral value after flood damage with partial pass-through.

    V_iv = V_i0 × (1 − θ × d)

    θ = 1.0 : full pass-through — £1 of damage reduces collateral by £1
    θ < 1.0 : partial pass-through — insurance, resilience investment, etc.

    Parameters
    ----------
    v0 : array-like, shape (n_properties,)
        Property value at origination (£m).
    d : array-like, shape (n_simulations, n_properties) or (n_properties,)
        Damage ratio. Broadcast against v0.
    theta : float
        Damage pass-through parameter ∈ (0, 1].

    Returns
    -------
    np.ndarray
        Impaired collateral value. Clipped at v0 × 1e-6 to avoid division by
        zero in downstream LTV computation.
    """
    v0 = np.asarray(v0, dtype=float)
    d  = np.asarray(d,  dtype=float)
    v_imp = v0 * (1.0 - theta * d)
    return np.maximum(v_imp, v0 * 1e-6)


def compute_ltv(loan: np.ndarray, v_impaired: np.ndarray) -> np.ndarray:
    """
    Current loan-to-value after damage.

    LTV_iv = Loan_i / V_iv

    Parameters
    ----------
    loan : array-like, shape (n_properties,)
        Outstanding loan balance (£m). Equal to EAD in the stylised model.
    v_impaired : array-like, shape (n_simulations, n_properties) or (n_properties,)
        Impaired collateral value from impaired_collateral_value(). Broadcast
        against loan.

    Returns
    -------
    np.ndarray of current LTV values.
    """
    loan      = np.asarray(loan,      dtype=float)
    v_impaired = np.asarray(v_impaired, dtype=float)
    return loan / v_impaired


def classify_outcome(
    ltv: np.ndarray,
    ltv_bar_1: float,
    ltv_bar_2: float,
) -> np.ndarray:
    """
    Classify each loan into a Basel III SA risk-weight bucket.

    Outcome codes:
        0 — Standard     (LTV ≤ LTV̄₁)
        1 — Conditional  (LTV̄₁ < LTV ≤ LTV̄₂)
        2 — Reject       (LTV > LTV̄₂)

    Parameters
    ----------
    ltv : array-like
        Current LTV (any shape — fully vectorised).
    ltv_bar_1 : float
        Lower LTV threshold (standard / conditional boundary).
    ltv_bar_2 : float
        Upper LTV threshold (conditional / reject boundary).

    Returns
    -------
    np.ndarray of int (same shape as ltv), values in {0, 1, 2}.
    """
    ltv = np.asarray(ltv, dtype=float)
    return np.where(ltv <= ltv_bar_1, 0,
           np.where(ltv <= ltv_bar_2, 1, 2)).astype(int)


def compute_rwa(
    loan: np.ndarray,
    outcome: np.ndarray,
    rw_standard: float,
    rw_conditional: float,
    rw_reject: float,
) -> np.ndarray:
    """
    Risk-weighted assets per loan.

    RWA_i = Loan_i × RW(outcome_i)

    Risk weights map:
        outcome = 0 → rw_standard
        outcome = 1 → rw_conditional
        outcome = 2 → rw_reject

    Parameters
    ----------
    loan : array-like, shape (n_properties,)
        Outstanding loan balance (£m).
    outcome : array-like, shape (n_simulations, n_properties) or (n_properties,)
        Integer outcome codes from classify_outcome(). Broadcast against loan.
    rw_standard, rw_conditional, rw_reject : float
        Risk weights from config["decision2_secured_lending"]["risk_weights"].

    Returns
    -------
    np.ndarray of RWA values (£m), same shape as outcome.
    """
    loan    = np.asarray(loan,    dtype=float)
    outcome = np.asarray(outcome, dtype=int)
    rw_map  = np.array([rw_standard, rw_conditional, rw_reject], dtype=float)
    return loan * rw_map[outcome]


def compute_capital(rwa: np.ndarray, capital_ratio: float) -> np.ndarray:
    """
    Minimum regulatory capital requirement.

    Capital = capital_ratio × RWA

    Parameters
    ----------
    rwa : array-like
        Risk-weighted assets (£m), any shape.
    capital_ratio : float
        Basel III minimum capital ratio (e.g. 0.08).

    Returns
    -------
    np.ndarray of capital requirement values (£m), same shape as rwa.
    """
    return np.asarray(rwa, dtype=float) * capital_ratio


def analytical_damage_thresholds(
    loan: np.ndarray,
    v0: np.ndarray,
    ltv_bar: float,
    theta: float,
) -> np.ndarray:
    """
    Damage ratio d* at which current LTV exactly reaches ltv_bar.

    Derived from LTV_iv = LTV̄:
        Loan / (V_i0 × (1 − θ × d*)) = ltv_bar
        ⟹  d* = (1 − Loan / (V_i0 × ltv_bar)) / θ

    Note: this is the exact inversion of impaired_collateral_value() +
    compute_ltv(). Equivalently, d* = (1 − LTV_i0 / ltv_bar) / θ, where
    LTV_i0 = Loan / V_i0 is the origination LTV.

    Parameters
    ----------
    loan : array-like, shape (n_properties,)
        Outstanding loan balance (£m).
    v0 : array-like, shape (n_properties,)
        Property value at origination (£m).
    ltv_bar : float
        LTV threshold to solve for (e.g. ltv_bar_1 or ltv_bar_2).
    theta : float
        Damage pass-through parameter.

    Returns
    -------
    np.ndarray of shape (n_properties,).
        d* per asset. Returns NaN where the threshold is unreachable
        within d ∈ [0, 1) — i.e. where the baseline LTV already exceeds
        ltv_bar (d* < 0) or where full damage cannot push LTV to ltv_bar
        even with θ = 1 (d* ≥ 1).
    """
    loan = np.asarray(loan, dtype=float)
    v0   = np.asarray(v0,   dtype=float)

    ltv_0 = loan / v0                          # origination LTV
    d_star = (1.0 - ltv_0 / ltv_bar) / theta   # exact inversion

    # Return NaN for unreachable thresholds
    d_star = np.where((d_star >= 0.0) & (d_star < 1.0), d_star, np.nan)
    return d_star
