# Climate Risk Model Uncertainty & Financial Decision-Making

## Research Question
How does uncertainty in vendor physical climate risk models translate into material differences
in financial outcomes — and what does that imply for how financial institutions should treat model choice?

## Data Source
CFRF/GARP Benchmarking Study (2025): 13 vendors, 100 properties, defended combined flooding at RP200 in 2030.
Extracted from Figures 19 (flood depth) and 20 (damage ratios).

## Project Structure

```
climate-risk-uncertainty/
├── data/
│   ├── raw/                          # Original extracted data
│   └── processed/                    # Fitted distributions, constructed portfolios
├── notebooks/
│   ├── shared/                       # Shared across all stylised decisions
│   │   ├── 01_vendor_uncertainty_distributions.ipynb
│   │   └── 02_portfolio_construction.ipynb
│   ├── decision1_credit_risk/        # IFRS 9 ECL staging
│   ├── decision2/                    # TBD
│   ├── decision3/                    # TBD
│   └── decision4/                    # TBD
├── src/
│   ├── uncertainty/                  # Distribution fitting and sampling
│   ├── portfolio/                    # Portfolio construction and loan characteristics
│   └── utils/                        # Plotting, I/O helpers
├── outputs/
│   ├── figures/
│   └── tables/
└── config/
    └── parameters.yaml               # All scenario parameters
```

## Stylised Decisions
1. **Credit risk / IFRS 9 provisioning**: Vendor uncertainty → PD/LGD → SICR staging → ECL
2. TBD
3. TBD
4. TBD

## Dependencies
numpy, pandas, scipy, matplotlib, seaborn, pyyaml, jupyter
