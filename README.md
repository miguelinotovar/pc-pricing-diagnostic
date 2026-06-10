# P&C Portfolio Monitoring & Pricing Diagnostic

A reproducible actuarial analytics workflow for a synthetic personal auto insurance portfolio.

The project demonstrates how policy-level data can be converted into portfolio experience diagnostics, frequency model support, observed-versus-expected review, monitoring signals, Excel review packs, and executive-facing reporting.

The main business use case is **portfolio monitoring and early warning**: identifying which segments should be escalated, investigated, monitored, or left without immediate action before committing to a full pricing or underwriting review.

This is a technical portfolio project built with synthetic data. It is not insurance advice, a production pricing model, or a final actuarial rate indication.

## Commercial sample

The project includes business-facing materials that translate the technical workflow into reviewable outputs:

* [Portfolio Monitoring One-Pager](commercial/pc_pricing_diagnostic_one_pager.pdf)
* [Executive Memo](commercial/pc_pricing_diagnostic_executive_memo.pdf)

HTML versions are also available in `commercial/`.

## What this project demonstrates

This repository is intended to show practical remote actuarial analytics support capabilities, including:

* synthetic P&C policy-level data generation;
* portfolio experience review by segment;
* frequency, loss ratio, pure premium, and observed-versus-expected diagnostics;
* benchmark Poisson frequency modeling with exposure offsets;
* model specification comparison and calibration checks;
* residual diagnostics and model review exhibits;
* portfolio monitoring watchlist construction;
* Excel review packs, chart outputs, and executive reporting materials.

The broader value is not the pricing model alone. The value is the full workflow:

```text
visible technical output
+ reproducible analytics process
+ business interpretation
+ remote actuarial support
```

## Remote actuarial support use cases

The workflow is designed to resemble delegated analytics work that could support a pricing, underwriting, actuarial, MGA, program business, or insurance analytics team.

Examples of support this project is meant to demonstrate:

* preparing recurring portfolio monitoring exhibits;
* identifying high-frequency, high-loss-ratio, or adverse O/E segments;
* producing Excel review packs for pricing or underwriting discussion;
* organizing model diagnostics into business-readable outputs;
* translating technical signals into escalation, investigation, and monitoring priorities;
* preparing executive summaries from actuarial analytics workflows.

## Workflow

The full pipeline follows this sequence:

```text
Synthetic data
→ Experience diagnostics
→ Frequency analysis
→ Portfolio monitoring watchlist
→ Executive reporting
```

Run the full workflow from the repository root:

```bash
PYTHONPATH=src python -m pc_pricing_diagnostic.pipeline
```

## Repository structure

```text
pc-pricing-diagnostic/
├── commercial/
│   ├── one_pager.html
│   ├── executive_memo.html
│   ├── pc_pricing_diagnostic_one_pager.pdf
│   ├── pc_pricing_diagnostic_executive_memo.pdf
│   └── styles.css
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── excel/
│   ├── figures/
│   └── tables/
├── src/
│   └── pc_pricing_diagnostic/
│       ├── synthetic_data.py
│       ├── experience_diagnostics.py
│       ├── frequency_analysis.py
│       ├── portfolio_monitoring.py
│       ├── executive_reporting.py
│       ├── pipeline.py
│       ├── config.py
│       ├── io_utils.py
│       └── plot_style.py
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Core modules

### `synthetic_data.py`

Generates the synthetic personal auto portfolio used throughout the project.

Main outputs:

```text
data/processed/synthetic_policy_data.csv
outputs/tables/synthetic_portfolio_summary.csv
```

### `experience_diagnostics.py`

Creates the first layer of portfolio review before modeling.

It produces portfolio-level summaries, segment experience tables, frequency indices, loss ratio indices, pure premium diagnostics, and exploratory frequency exhibits.

Main outputs include:

```text
outputs/tables/portfolio_overview.csv
outputs/tables/segment_experience_by_territory_age.csv
outputs/tables/segment_experience_by_vehicle_age.csv
outputs/tables/segment_experience_by_territory_vehicle.csv
outputs/tables/high_frequency_segments.csv
outputs/excel/experience_review.xlsx
outputs/excel/exploratory_frequency_review.xlsx
```

### `frequency_analysis.py`

Consolidates the actuarial frequency modeling workflow.

It includes a benchmark Poisson GLM with log exposure offset, model comparison, observed-versus-expected claim diagnostics, calibration by expected-risk decile, residual diagnostics, and model specification comparison.

Main outputs include:

```text
outputs/tables/frequency_model_comparison.csv
outputs/tables/frequency_model_coefficients.csv
outputs/tables/frequency_oe_by_territory_age.csv
outputs/tables/frequency_oe_by_territory_vehicle.csv
outputs/tables/frequency_oe_by_vehicle_age.csv
outputs/tables/frequency_model_diagnostics_summary.csv
outputs/tables/frequency_model_calibration_by_decile.csv
outputs/excel/frequency_model_review.xlsx
outputs/excel/frequency_model_specification_comparison.xlsx
outputs/excel/frequency_model_diagnostics.xlsx
```

### `portfolio_monitoring.py`

Creates the portfolio monitoring and early-warning layer.

This module combines experience diagnostics, observed-versus-expected results, credibility/volume signals, and technical pressure indicators into a monitoring watchlist.

Segments are classified into:

```text
Escalate
Investigate
Monitor
No immediate action
```

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

These outputs support portfolio review conversations, model review discussions, and business-facing summaries.

Main outputs:

```text
outputs/figures/executive_top_frequency_segments.png
outputs/figures/executive_top_loss_ratio_segments.png
outputs/figures/executive_calibration_by_decile.png
outputs/figures/executive_model_specification_comparison.png
outputs/tables/executive_visual_inventory.csv
outputs/excel/executive_visual_pack.xlsx
```

## Technical notes

The frequency modeling component uses a Poisson GLM with exposure offsets to estimate expected claim counts. Observed-versus-expected ratios are used as diagnostic signals, not as standalone pricing recommendations.

The monitoring layer combines multiple signals:

* observed claim frequency;
* loss ratio pressure;
* frequency index;
* loss ratio index;
* observed-versus-expected claim behavior;
* segment volume and credibility indicators.

The output is a prioritization framework for review, not an automatic pricing action.

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
PYTHONPATH=src python -m pc_pricing_diagnostic.pipeline
```

## Key outputs

The project produces three main types of outputs:

```text
outputs/tables/   CSV tables for review and auditability
outputs/excel/    Excel review packs for business users
outputs/figures/  Charts for reporting and executive summaries
```

The commercial materials are stored in:

```text
commercial/
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
