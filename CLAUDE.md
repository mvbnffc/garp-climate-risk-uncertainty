# CLAUDE.md — Project Context for Claude Code

## What this project is

A research project developing a **simulation framework for propagating physical climate model uncertainty through stylised financial decisions**. The core question: when financial institutions buy physical climate risk data from commercial vendors, those vendors disagree — sometimes dramatically — on the hazard and loss estimates for the same property. How does that vendor disagreement translate into materially different financial outcomes?

This is being developed for **academic publication** (likely targeting a climate finance or environmental economics journal) and **practitioner application** (aimed at financial institutions and regulators). The principal investigator has dual affiliations at LSE Grantham and Oxford ECI/OPSIS.

## Data

The empirical basis is the **CFRF/GARP Benchmarking Study (2025)**, which compared 13 commercial physical climate risk vendors using a portfolio of 100 UK properties. We extracted summary statistics (minimum, mean, maximum across vendors) for:
- Flood depth (metres) — Figure 19
- Damage ratio (0–1) — Figure 20

These are for **defended combined (pluvial + fluvial) flooding at 1-in-200 year return period in 2030**.

The data is in `data/raw/cfrf_garp_defended_flood.csv`. Note:
- Column `minimim_dr` is a typo for `minimum_dr` — clean on load
- There's a trailing empty row — drop rows where `property_rank` is NaN
- 20 properties have zero damage across all vendors
- 5 properties have mean_dr=0 but max_dr=0.01 (one vendor reports marginal damage)
- The remaining ~75 properties have meaningful vendor uncertainty

## Project structure

```
notebooks/shared/          — Infrastructure used by ALL stylised decisions
notebooks/decision1_credit_risk/  — IFRS 9 ECL staging simulation
notebooks/decision2-4/     — TBD (other financial decisions)
src/uncertainty/           — Distribution fitting and sampling
src/portfolio/             — Portfolio construction (to be built)
src/utils/                 — Plotting, I/O helpers
config/parameters.yaml     — ALL scenario parameters (single source of truth)
outputs/figures/shared/    — Figures from shared notebooks
outputs/figures/decision1/ — Figures from Decision 1
```

## The four stylised decisions (simulation stages)

1. **Credit risk / IFRS 9 provisioning** (in progress): Vendor uncertainty in damage ratios → property value → LTV → PD/LGD → SICR staging → Expected Credit Loss. The key mechanism is that IFRS 9 staging creates a **discontinuity**: a small difference in vendor damage estimates can flip an exposure from Stage 1 (12-month ECL) to Stage 2 (lifetime ECL), causing a large jump in recognised provisions.

2–4. **TBD** — other financial decisions where vendor model choice matters. The shared infrastructure (distribution fitting, portfolio construction) is designed to be reused across all four.

## Key methodological decisions already made

### Distribution fitting (Notebook 01)
- **Three distribution families**: Beta (primary), Triangular, Uniform (maximum-entropy / most dispersed)
- **Three quantile interpretations** for what vendor min/max represent:
  - `order_statistic`: min/max as expected extremes of 13 draws (quantiles at 1/14 and 13/14)
  - `quantile_approx`: min/max as approximate percentiles (1/(k+1) and k/(k+1))
  - `extremes`: min/max as true support bounds (widest interpretation)
- **Beta fitting uses mean-preserving quantile matching**: parameterise as a = μν, b = (1−μ)ν where μ is the observed mean, then optimise the concentration parameter ν to match min/max quantiles
- **Zero-damage properties** are excluded from fitting and re-added at portfolio stage as point masses at d=0

### Credit risk transmission (Decision 1) — TWO MODELS

#### Primary model: LTV-based structural transmission

This is grounded in the standard mortgage/secured lending credit risk literature. The single economic mechanism is: physical damage destroys property value, which increases the current loan-to-value ratio, which drives both PD and LGD through a structurally coherent channel.

**Key references:**
- Deng, Quigley & Van Order (2000), *Econometrica* — option-theoretic mortgage default model; default as put option exercise when LTV > 1
- Campbell & Cocco (2015), *J. Financial Economics* — structural model decomposing default as Pr(Equity < 0) × Pr(Default | Equity < 0)
- Qi & Yang (2009), *J. Banking & Finance* — current LTV as the single most important determinant of mortgage LGD
- Bhutta, Dokko & Shan (2017), *Rev. Financial Studies* — depth of negative equity needed to trigger default (equity shortfalls >50% before strategic default prevalent)
- Shleifer & Vishny (1992), *J. Finance* — theoretical foundation for fire-sale collateral discounts
- Lekkas, Quigley & Van Order (1993), *Real Estate Economics* — higher LTV → higher loss severity
- Ferretti et al. (2023), *J. Financial Econometrics* — logistic PD specification as standard industry practice
- Leow & Mues (2012), *Int. J. Forecasting* — LGD modelling for UK mortgage data

**Transmission chain:**
```
damage_ratio (d)
  → property value:  PV(d) = PV_0 × (1 − d)
  → current LTV:     LTV(d) = LTV_0 / (1 − d)
  → PD (logistic):   PD_12m(d) = 1 / (1 + exp(−(β₀ + β₁ × LTV(d))))
  → LGD (shortfall): LGD(d) = max(1 − (1−d)(1−ω) / LTV_0, 0)
  → lifetime PD:     PD_LT(d) = 1 − (1 − PD_12m(d))^T
  → SICR test:       PD_LT(d) / PD_LT_baseline > τ
  → ECL:             stage-dependent provision
```

**Key structural properties:**
- PD and LGD are driven by the **same state variable** (current LTV) — they are not independently parameterised
- Nonlinearity in PD as a function of d **emerges from the model structure** (the 1/(1−d) LTV transformation composed with the logistic) rather than being imposed by an arbitrary exponential
- Parameters (β₀, β₁, ω, LTV₀) are **observable or have well-documented empirical ranges**, unlike the ad hoc α and λ in the reduced-form model
- Loans with high baseline LTV are closer to the SICR boundary → more sensitive to vendor disagreement. This interaction between portfolio characteristics and vendor uncertainty is a key result the simulation should demonstrate.

**Default calibration targets (from parameters.yaml):**
- β₀ = −11.5, β₁ = 8.0 gives: LTV=0.70 → PD≈0.27%, LTV=0.90 → PD≈1.34%, LTV=1.10 → PD≈6.30%
- ω = 0.25 (25% workout/fire-sale discount)

#### Sensitivity model: Reduced-form (ad hoc) transmission

Kept as a robustness check. Two independent functions with free parameters:
- PD: `PD_12m = PD_0 × exp(α × d)` where α = 2.0
- LGD: `LGD = min(LGD_0 + λ × d, 1)` where λ = 0.25
- Baseline PD₀ and LGD₀ drawn independently from portfolio distributions

The purpose of retaining this model is to show that the qualitative results (vendor uncertainty causing staging discontinuities at the SICR threshold) hold regardless of the specific transmission model chosen.

### IFRS 9 staging and ECL (common to both models)
- SICR threshold: `PD_LT(d) / PD_LT_baseline > τ` where τ=2 is the EBA benchmark (doubling rule)
- Lifetime PD: `PD_LT = 1 − (1 − PD_12m)^T` (independence approximation, noted as simplification)
- ECL Stage 1: `PD_12m × LGD × EAD`; Stage 2: `PD_LT × LGD × EAD` (simplified; discounted marginal version available as sensitivity)
- Two correlation scenarios: independent vendor draws per property, or perfectly correlated (single quantile draw = choosing one vendor for the whole portfolio)

### Portfolio construction (Notebook 02, not yet built)
- 100 loans, one per CFRF property
- **In the LTV-based model**: each loan has LTV₀ (drawn uniform 0.60–0.90), property value PV₀ (lognormal), maturity T (uniform 5–25y). EAD = LTV₀ × PV₀. Baseline PD is derived from the logistic at LTV₀. Baseline LGD is derived from the collateral shortfall formula at d=0.
- **In the reduced-form model**: EAD, baseline PD, baseline LGD are drawn independently from distributions.
- Heterogeneity in LTV₀ matters because it determines how close each loan is to the SICR boundary.

## Code conventions

- **Python** is the implementation language. Standard scientific stack: numpy, pandas, scipy, matplotlib, seaborn
- `config/parameters.yaml` is the single source of truth for all scenario parameters — do not hardcode values in notebooks
- `src/` modules contain reusable functions; notebooks call these modules
- Figures are saved to `outputs/figures/{shared,decision1,...}/` at 300 DPI
- Processed data goes to `data/processed/`
- When running notebooks, the working directory is the notebook's own folder, so paths to config and data use `../../`

## Style and standards

- **Methodological rigour over simplicity** — this is for academic publication. Assumptions should be explicit and defensible.
- **Be direct about limitations and trade-offs** — if a modelling choice involves a trade-off, name it.
- **Notation consistency** — check `config/parameters.yaml` and existing notebooks before introducing new variable names. Key symbols:
  - d: damage ratio
  - LTV₀: loan-to-value at origination
  - LTV(d): current LTV after damage = LTV₀ / (1−d)
  - PV₀: property value at origination
  - β₀, β₁: logistic PD parameters
  - ω: workout/fire-sale discount
  - α, λ: reduced-form PD/LGD sensitivity parameters (sensitivity model only)
  - τ: SICR threshold
  - T: loan maturity
  - v: vendor index
  - i: property/loan index
- **Don't overcomplicate** — keep it as simple as possible while remaining grounded in what FIs might actually do. We dropped copulas from the Kerkhofs et al. framework deliberately.

## Reference papers

These were reviewed in developing the framework:
- **Kerkhofs et al. (2025)** "An asset-level analysis of financial tail risks under extreme weather events" — *Environ. Res.: Climate*. Our previous paper; full Monte Carlo with DCF valuation, copulas for spatial correlation. More complex than needed here but informed the modular design.
- **Pozdyshev, Lobanov & Ilinsky (2025)** "Incorporating physical climate risks into banks' credit risk models" — *BIS Working Paper 1274*. Extends the one-factor Vasicek/ASRF model with a physical risk jump factor (q-deformed normal distribution). Elegant but targets regulatory capital (RWA), not IFRS 9 staging. May inform a later stylised decision.
- **CFRF/GARP (2025)** "Risk Professional's Guide to Physical Risk Assessments" — the benchmarking study from which our data is extracted.

**Credit risk transmission references (for the LTV-based model):**
- Deng, Quigley & Van Order (2000), *Econometrica* 68(2) — option-theoretic mortgage default
- Campbell & Cocco (2015), *J. Financial Economics* — structural mortgage default model
- Qi & Yang (2009), *J. Banking & Finance* 33(5) — LTV as primary LGD determinant
- Bhutta, Dokko & Shan (2017), *Rev. Financial Studies* — depth of negative equity
- Shleifer & Vishny (1992), *J. Finance* 47(4) — fire-sale discounts
- Lekkas, Quigley & Van Order (1993), *Real Estate Economics* 21(4) — LTV and loss severity
- Ferretti et al. (2023), *J. Financial Econometrics* 21(2) — logistic PD specification
- Leow & Mues (2012), *Int. J. Forecasting* 28(1) — LGD for UK mortgages

## What's been built so far

- [x] Directory structure
- [x] `config/parameters.yaml` — full configuration (updated with LTV-based model)
- [x] `src/uncertainty/distributions.py` — Beta, Triangular, Uniform fitting + sampling
- [x] `src/utils/plotting.py` — consistent figure style
- [x] `notebooks/shared/01_vendor_uncertainty_distributions.ipynb` — distribution fitting notebook
- [ ] `notebooks/shared/02_portfolio_construction.ipynb`
- [ ] `notebooks/decision1_credit_risk/03_credit_risk_mapping.ipynb`
- [ ] `notebooks/decision1_credit_risk/04_simulation_ecl_staging.ipynb`
- [ ] `notebooks/decision1_credit_risk/05_sensitivity_analysis.ipynb`

## Dependencies

```
numpy pandas scipy matplotlib seaborn pyyaml jupyter
```
