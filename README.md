# P&C Portfolio Monitoring & Pricing Diagnostic

This repository is a reproducible actuarial analytics workflow for a synthetic personal auto portfolio.

It demonstrates how to move from raw policy-period data to portfolio diagnostics, frequency modeling, monitoring signals, and executive reporting. The workflow is designed to show practical actuarial support capabilities for pricing, underwriting, portfolio review, and remote analytics work.

The project is not intended to provide insurance advice or production-ready pricing recommendations. It is a technical portfolio project built with synthetic data.

## What this project demonstrates

The workflow shows how to:

* generate a synthetic P&C insurance portfolio;
* review portfolio experience by segment;
* analyze frequency, loss ratio, pure premium, and observed-to-expected behavior;
* fit and compare Poisson frequency models with exposure offsets;
* evaluate model calibration and residual diagnostics;
* convert technical signals into a portfolio monitoring watchlist;
* produce executive-ready tables, charts, Excel packs, and commercial materials.

The strongest business use case is portfolio monitoring and early warning: identifying which segments should be escalated, investigated, monitored, or left without immediate action.

## Workflow

The pipeline follows this sequence:

```text
Synthetic data
→ Experience diagnostics
→ Frequency analysis
→ Portfolio monitoring watchlist
→ Executive reporting
```

Run the full pipeline from the repository root:

```bash
PYTHONPATH=src python -m pc_pricing_diagnostic.pipeline
```

## Repository structure

```text
pc-pricing-diagnostic/
├── commercial/
│   ├── executive_memo.html
│   ├── one_pager.html
│   ├── pc_pricing_diagnostic_executive_memo.pdf
│   ├── pc_pricing_diagnostic_one_pager.pdf
│   └── styles.css
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
├── outputs/
│   ├── excel/
│   ├── figures/
│   └── tables/
├── src/
│   └── pc_pricing_diagnostic/
│       ├── __init__.py
│       ├── config.py
│       ├── executive_reporting.py
│       ├── experience_diagnostics.py
│       ├── frequency_analysis.py
│       ├── io_utils.py
│       ├── pipeline.py
│       ├── plot_style.py
│       ├── portfolio_monitoring.py
│       └── synthetic_data.py
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Business modules

### `synthetic_data.py`

Generates the synthetic personal auto portfolio used throughout the project.

The output includes policy-period records with exposure, claim count, earned premium, total claim amount, territory, driver age, vehicle age, vehicle type, and derived rating bands.

Main outputs:

```text
data/processed/synthetic_policy_data.csv
outputs/tables/synthetic_portfolio_summary.csv
```

### `experience_diagnostics.py`

Creates the first layer of portfolio review before modeling.

It produces:

* portfolio-level overview;
* segment experience tables;
* frequency and loss ratio indices;
* high-frequency segment review;
* exploratory diagnostics by driver age, vehicle age, territory, and vehicle type;
* modeling rationale for benchmark rating variables.

Main outputs:

```text
outputs/tables/portfolio_overview.csv
outputs/tables/segment_experience_by_territory_age.csv
outputs/tables/segment_experience_by_vehicle_age.csv
outputs/tables/segment_experience_by_territory_vehicle.csv
outputs/tables/high_frequency_segments.csv
outputs/tables/eda_frequency_by_driver_age.csv
outputs/tables/eda_frequency_by_driver_age_band.csv
outputs/tables/eda_frequency_by_vehicle_age.csv
outputs/tables/eda_frequency_by_vehicle_age_band.csv
outputs/tables/eda_frequency_by_territory.csv
outputs/tables/eda_frequency_by_vehicle_type.csv
outputs/tables/eda_modeling_rationale.csv
outputs/excel/experience_review.xlsx
outputs/excel/exploratory_frequency_review.xlsx
```

### `frequency_analysis.py`

Consolidates the actuarial frequency modeling workflow.

It includes:

* Poisson GLM with log exposure offset;
* null model comparison;
* benchmark categorical frequency model;
* observed-to-expected tables by segment;
* model specification comparison;
* out-of-sample mean Poisson deviance ranking;
* calibration by expected-risk decile;
* residual diagnostics;
* randomized quantile residuals;
* diagnostic rationale.

Main outputs:

```text
outputs/tables/frequency_model_comparison.csv
outputs/tables/frequency_model_coefficients.csv
outputs/tables/frequency_oe_by_territory_age.csv
outputs/tables/frequency_oe_by_territory_vehicle.csv
outputs/tables/frequency_oe_by_vehicle_age.csv
outputs/tables/frequency_model_specification_comparison.csv
outputs/tables/frequency_model_specification_ranking.csv
outputs/tables/frequency_model_specification_rationale.csv
outputs/tables/frequency_model_diagnostics_summary.csv
outputs/tables/frequency_model_calibration_by_decile.csv
outputs/tables/frequency_model_top_residuals.csv
outputs/tables/frequency_model_diagnostics_rationale.csv
outputs/excel/frequency_model_review.xlsx
outputs/excel/frequency_model_specification_comparison.xlsx
outputs/excel/frequency_model_diagnostics.xlsx
```

### `portfolio_monitoring.py`

Creates the portfolio monitoring and early-warning layer.

This module translates technical signals into an actionable watchlist. Segments are classified into:

```text
Escalate
Investigate
Monitor
No immediate action
```

The watchlist is designed to support recurring portfolio review, pricing discussion, underwriting review, and remote actuarial analytics support.

Main outputs:

```text
outputs/tables/portfolio_monitoring_watchlist.csv
outputs/tables/portfolio_monitoring_summary.csv
outputs/tables/portfolio_monitoring_rationale.csv
outputs/excel/portfolio_monitoring_review.xlsx
outputs/figures/portfolio_warning_matrix.png
outputs/figures/portfolio_monitoring_top_priorities.png
```

### `executive_reporting.py`

Produces executive-oriented visual outputs and an Excel visual pack.

These outputs are intended to support commercial reporting, portfolio review conversations, and client-facing summaries.

Main outputs:

```text
outputs/figures/executive_top_frequency_segments.png
outputs/figures/executive_top_loss_ratio_segments.png
outputs/figures/executive_calibration_by_decile.png
outputs/figures/executive_model_specification_comparison.png
outputs/tables/executive_visual_inventory.csv
outputs/excel/executive_visual_pack.xlsx
```

## Support modules

### `pipeline.py`

Runs the full workflow in sequence.

```bash
PYTHONPATH=src python -m pc_pricing_diagnostic.pipeline
```

### `config.py`

Centralizes repository paths for data, outputs, figures, Excel files, and commercial materials.

### `io_utils.py`

Provides reusable input/output helpers for reading required tables and writing CSV outputs.

### `plot_style.py`

Centralizes chart styling, color palette, warning-level colors, axis formatting, and figure export settings.

## Commercial materials

The commercial materials are stored in:

```text
commercial/
```

Current files include:

```text
commercial/one_pager.html
commercial/executive_memo.html
commercial/pc_pricing_diagnostic_one_pager.pdf
commercial/pc_pricing_diagnostic_executive_memo.pdf
commercial/styles.css
```

These materials should be read as portfolio artifacts, not production insurance advice. They demonstrate how the technical workflow can be translated into business-facing reporting.

## Current positioning

The project is best described as:

```text
P&C Portfolio Monitoring & Pricing Diagnostic
```

The pricing model is one technical component. The broader value is the full analytics workflow:

```text
visible technical output
+ reproducible analytics workflow
+ business interpretation
+ remote actuarial support
```

## Intended audience

This project is intended for:

* actuarial analytics portfolio review;
* P&C pricing and monitoring discussions;
* remote actuarial support positioning;
* demonstrations of Python-based insurance analytics;
* examples of turning technical diagnostics into business-facing outputs.

## Limitations

The data is synthetic.

The workflow is designed for demonstration and portfolio purposes. It does not replace production actuarial pricing, regulatory review, underwriting judgment, reserving analysis, or insurance company governance.

The outputs are not recommendations to change rates, underwriting rules, or business strategy. They are diagnostic signals for further review.
