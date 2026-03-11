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

1. **Credit risk / IFRS 9 provisioning** (in progress): Vendor uncertainty in damage ratios → PD/LGD adjustments → SICR staging threshold → Expected Credit Loss. The key mechanism is that IFRS 9 staging creates a **discontinuity**: a small difference in vendor damage estimates can flip an exposure from Stage 1 (12-month ECL) to Stage 2 (lifetime ECL), causing a large jump in recognised provisions.

2–4. **TBD** — other financial decisions where vendor model choice matters. The shared infrastructure (distribution fitting, portfolio construction) is designed to be reused across all four.

## Key methodological decisions already made

### Distribution fitting (Notebook 01)
- **Three distribution families**: Triangular (primary), Uniform (maximum-entropy / most dispersed), Beta (robustness comparison only)
- **Three quantile interpretations** for what vendor min/max represent (relevant to Beta only):
  - `order_statistic`: min/max as expected extremes of 13 draws (quantiles at 1/14 and 13/14)
  - `quantile_approx`: min/max as approximate percentiles (1/(k+1) and k/(k+1))
  - `extremes`: min/max as true support bounds (widest interpretation)
- **Beta fitting uses mean-preserving quantile matching**: parameterise as a = μν, b = (1−μ)ν where μ is the observed mean, then optimise the concentration parameter ν to match min/max quantiles
- **Zero-damage properties** are excluded from fitting and re-added at portfolio stage as point masses at d=0

### Credit risk mapping (Decision 1)
- Transmission chain: `damage_ratio → PD, LGD → SICR test → ECL`
- PD mapping: `PD_12m = PD_0 * exp(α * d)` where α is a sensitivity parameter
- LGD mapping: `LGD = min(LGD_0 + λ * d, 1)` where λ is a sensitivity parameter
- Lifetime PD: `PD_LT = 1 − (1 − PD_12m)^T` (independence approximation, noted as simplification)
- SICR threshold: `PD_LT / PD_0_LT > τ` where τ=2 is the EBA benchmark (doubling rule)
- ECL Stage 1: `PD_12m × LGD × EAD`; Stage 2: `PD_LT × LGD × EAD` (simplified; discounted marginal version available as sensitivity)
- Two correlation scenarios: independent vendor draws per property, or perfectly correlated (single quantile draw = choosing one vendor for the whole portfolio)

### Portfolio construction (Notebook 02, not yet built)
- 100 loans, one per CFRF property
- Heterogeneous characteristics drawn from distributions (see `config/parameters.yaml`): EAD (£0.5m–5m), maturity (5–25y), baseline PD (lognormal, median ~0.4%), baseline LGD (20%–45%)
- Heterogeneity matters because the SICR threshold interacts with baseline credit quality

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
- **Notation consistency** — check `config/parameters.yaml` and existing notebooks before introducing new variable names. Key symbols: d (damage ratio), α (PD sensitivity), λ (LGD sensitivity), τ (SICR threshold), v (vendor index), i (property/loan index).
- **Don't overcomplicate** — keep it as simple as possible while remaining grounded in what FIs might actually do. We dropped copulas from the Kerkhofs et al. framework deliberately.

## Reference papers

These were reviewed in developing the framework:
- **Kerkhofs et al. (2025)** "An asset-level analysis of financial tail risks under extreme weather events" — *Environ. Res.: Climate*. Our previous paper; full Monte Carlo with DCF valuation, copulas for spatial correlation. More complex than needed here but informed the modular design.
- **Pozdyshev, Lobanov & Ilinsky (2025)** "Incorporating physical climate risks into banks' credit risk models" — *BIS Working Paper 1274*. Extends the one-factor Vasicek/ASRF model with a physical risk jump factor (q-deformed normal distribution). Elegant but targets regulatory capital (RWA), not IFRS 9 staging. May inform a later stylised decision.
- **CFRF/GARP (2025)** "Risk Professional's Guide to Physical Risk Assessments" — the benchmarking study from which our data is extracted.

## What's been built so far

- [x] Directory structure
- [x] `config/parameters.yaml` — full configuration
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
