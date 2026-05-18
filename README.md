# P&C Pricing Diagnostic

A synthetic P&C actuarial analytics project that demonstrates a compact pricing diagnostic workflow for personal auto insurance.

The project is designed to identify segments that may require pricing, underwriting, risk classification, model calibration, or data quality review before committing to a full pricing study.

This repository combines:

- synthetic portfolio generation;
- experience review;
- exploratory frequency diagnostics;
- benchmark Poisson frequency modeling;
- model specification comparison;
- model diagnostics;
- executive visuals;
- commercial one-pager and executive memo examples.

## Purpose

The core business question is:

> Which segments appear to require pricing, underwriting, or data quality review, and is there enough technical support to prioritize the next action?

The project is not intended to produce final rate indications. It is a diagnostic workflow that helps prioritize where deeper pricing or underwriting work may be valuable.

## Repository structure

```text
pc-pricing-diagnostic/
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
├── outputs/
│   ├── excel/
│   ├── figures/
│   └── tables/
├── reports/
│   └── commercial/
│       ├── executive_memo.html
│       ├── one_pager.html
│       ├── pc_pricing_diagnostic_executive_memo.pdf
│       ├── pc_pricing_diagnostic_one_pager.pdf
│       └── styles.css
├── scripts/
│   └── run_pipeline.py
├── src/
│   └── pc_pricing_diagnostic/
├── README.md
├── requirements.txt
└── pyproject.toml

## Main commercial outputs

The commercial-facing sample deliverables are in:

```text
reports/commercial/
```

Key files:

```text
reports/commercial/pc_pricing_diagnostic_one_pager.pdf
reports/commercial/pc_pricing_diagnostic_executive_memo.pdf
reports/commercial/one_pager.html
reports/commercial/executive_memo.html
reports/commercial/styles.css
```

The PDFs are designed for prospect conversations. The HTML/CSS files provide reusable commercial templates for the one-pager and executive memo.

## Main technical outputs

Generated technical outputs are organized into three folders:

```text
outputs/tables/
outputs/figures/
outputs/excel/
```

Examples of generated outputs include:

- portfolio overview tables;
- segment experience tables;
- high-frequency segment diagnostics;
- exploratory frequency tables and charts;
- benchmark frequency model outputs;
- observed-versus-expected diagnostics;
- model specification ranking;
- calibration by expected-risk decile;
- residual diagnostics;
- executive visual inventory;
- Excel review packs.

## Methodological overview

The workflow follows these stages.

### 1. Synthetic data generation

The project starts with a synthetic personal auto portfolio containing policy-level exposure, premium, claim count, claim amount, and rating variables.

### 2. Experience review

The experience review summarizes claim frequency, claim amount, earned premium, loss ratio, pure premium, and frequency index by selected segments.

### 3. Exploratory frequency diagnostics

The exploratory layer reviews observed frequency patterns by driver age, vehicle age, territory, vehicle type, and grouped rating variables.

This step supports decisions about whether variables should enter a first benchmark model as categorical, continuous, or candidates for later nonlinear treatment.

### 4. Benchmark frequency model

The benchmark model is a Poisson GLM for claim counts with a log exposure offset.

The model estimates expected claim counts by policy-period and supports observed-versus-expected analysis across segments.

### 5. Model specification comparison

Alternative frequency model specifications are compared using held-out test mean Poisson deviance.

The comparison is used to support model selection, but not mechanically. Interpretability, stability, credibility, and business usability also matter.

### 6. Model diagnostics

The diagnostic layer includes:

- calibration by expected-risk decile;
- observed-versus-expected comparisons;
- dispersion diagnostics;
- residual diagnostics;
- randomized quantile residual plots.

### 7. Executive visuals and commercial reporting

The final layer converts technical outputs into executive visuals and commercial-facing report templates.

## How to run the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full technical pipeline:

```bash
python scripts/run_pipeline.py
```

The pipeline regenerates the synthetic data, technical outputs, model diagnostics, and executive visuals.

Commercial PDFs are exported manually from the HTML files in `reports/commercial/` after visual review.

## Key scripts

```text
src/pc_pricing_diagnostic/synthetic_data.py
```

Generates synthetic personal auto policy data.

```text
src/pc_pricing_diagnostic/experience_review.py
```

Creates portfolio and segment-level experience review outputs.

```text
src/pc_pricing_diagnostic/exploratory_frequency.py
```

Creates exploratory claim frequency diagnostics and visual outputs.

```text
src/pc_pricing_diagnostic/frequency_model.py
```

Fits the benchmark Poisson frequency GLM and creates model outputs.

```text
src/pc_pricing_diagnostic/model_specification_comparison.py
```

Compares alternative model specifications.

```text
src/pc_pricing_diagnostic/frequency_model_diagnostics.py
```

Creates calibration, residual, and model diagnostic outputs.

```text
src/pc_pricing_diagnostic/executive_visuals.py
```

Creates executive charts and an Excel visual review pack.

```text
scripts/run_pipeline.py
```

Runs the full technical pipeline in order.

## Intended audience

This repository is intended for:

- pricing actuaries;
- pricing managers;
- underwriting managers;
- insurance analytics teams;
- MGAs and program administrators;
- consultants reviewing book performance;
- hiring managers evaluating applied actuarial analytics work.

## What this project demonstrates

This project demonstrates the ability to:

- structure an actuarial pricing diagnostic workflow;
- generate and validate synthetic insurance data;
- perform segment-level experience review;
- fit and evaluate Poisson frequency GLMs;
- compare model specifications;
- evaluate model calibration;
- produce observed-versus-expected diagnostics;
- translate technical results into executive visuals;
- create reusable commercial reporting templates.

## Limitations

This project uses synthetic data.

It is not a final actuarial pricing indication, rate filing, or substitute for actuarial judgment.

The workflow does not include:

- final rate indications;
- severity modeling;
- expense provisions;
- reinsurance effects;
- cost of capital;
- regulatory filing constraints;
- competitive positioning;
- credibility-weighted implementation;
- underwriting judgment;
- temporal trend analysis.

A production engagement would require validation of real policy, premium, exposure, claim, underwriting, and business-rule data.

## Suggested next step for a real portfolio

Use this diagnostic structure on one book, program, territory cluster, or segment group.

The practical objective would be to identify 3–5 pricing, underwriting, risk classification, or monitoring priorities supported by data, model diagnostics, and business interpretation.