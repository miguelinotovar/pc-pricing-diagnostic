from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm, poisson
from sklearn.model_selection import train_test_split

from pc_pricing_diagnostic.config import (
    OUTPUT_EXCEL,
    OUTPUT_FIGURES,
    OUTPUT_TABLES,
    PROCESSED_DATA,
)


DATA_PATH = PROCESSED_DATA / "synthetic_policy_data.csv"

SPECIFICATION_EXCEL_PATH = (
    OUTPUT_EXCEL / "frequency_model_specification_comparison.xlsx"
)
DIAGNOSTICS_EXCEL_PATH = OUTPUT_EXCEL / "frequency_model_diagnostics.xlsx"


MODEL_FORMULA = (
    'claim_count ~ '
    'C(territory, Treatment(reference="Rural")) + '
    'C(driver_age_band, Treatment(reference="40-64")) + '
    'C(vehicle_age_band, Treatment(reference="4-7")) + '
    'C(vehicle_type, Treatment(reference="Sedan"))'
)

NULL_FORMULA = "claim_count ~ 1"

CATEGORICAL_FORMULA = MODEL_FORMULA

CONTINUOUS_LINEAR_FORMULA = (
    'claim_count ~ '
    'C(territory, Treatment(reference="Rural")) + '
    'driver_age_centered + vehicle_age_centered + '
    'C(vehicle_type, Treatment(reference="Sedan"))'
)

CONTINUOUS_QUADRATIC_FORMULA = (
    'claim_count ~ '
    'C(territory, Treatment(reference="Rural")) + '
    'driver_age_centered + I(driver_age_centered ** 2) + '
    'vehicle_age_centered + I(vehicle_age_centered ** 2) + '
    'C(vehicle_type, Treatment(reference="Sedan"))'
)

HYBRID_FORMULA = (
    'claim_count ~ '
    'C(territory, Treatment(reference="Rural")) + '
    'C(driver_age_band, Treatment(reference="40-64")) + '
    'vehicle_age_centered + I(vehicle_age_centered ** 2) + '
    'C(vehicle_type, Treatment(reference="Sedan"))'
)

MODEL_SPECS = {
    "null_offset_only": NULL_FORMULA,
    "categorical_age_bands": CATEGORICAL_FORMULA,
    "continuous_linear_age": CONTINUOUS_LINEAR_FORMULA,
    "continuous_quadratic_age": CONTINUOUS_QUADRATIC_FORMULA,
    "hybrid_driver_band_vehicle_quadratic": HYBRID_FORMULA,
}


def load_policy_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the policy-period dataset used for frequency modeling.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Policy data not found at {path}. "
            "Run synthetic_data.py before running the frequency analysis."
        )

    df = pd.read_csv(path)

    required_columns = {
        "policy_id",
        "exposure",
        "claim_count",
        "territory",
        "driver_age",
        "driver_age_band",
        "vehicle_age",
        "vehicle_age_band",
        "vehicle_type",
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if (df["exposure"] <= 0).any():
        raise ValueError("Exposure must be strictly positive for log-offset modeling.")

    return df


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    """
    claim_indicator = (df["claim_count"] > 0).astype(int)

    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=claim_indicator,
    )

    return train.copy(), test.copy()


def fit_poisson_glm(formula: str, df: pd.DataFrame):
    """
    Fit a Poisson GLM with log(exposure) as an offset.
    """
    offset = np.log(df["exposure"])

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Poisson(),
        offset=offset,
    )

    return model.fit()


def compute_expected_claims(fitted_model, df: pd.DataFrame) -> np.ndarray:
    """
    Compute expected claim counts for each policy-period.
    """
    offset = np.log(df["exposure"])
    expected_claims = fitted_model.predict(df, offset=offset)

    return np.asarray(expected_claims).clip(min=1e-12)


def poisson_deviance(
    observed_counts: np.ndarray,
    expected_counts: np.ndarray,
) -> float:
    """
    Compute Poisson deviance.
    """
    observed_counts = np.asarray(observed_counts, dtype=float)
    expected_counts = np.asarray(expected_counts, dtype=float).clip(min=1e-12)

    term = np.zeros_like(observed_counts, dtype=float)

    positive = observed_counts > 0
    term[positive] = observed_counts[positive] * np.log(
        observed_counts[positive] / expected_counts[positive]
    )

    term -= observed_counts - expected_counts

    return float(2 * term.sum())


def create_model_comparison(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fitted_models: dict[str, object],
) -> pd.DataFrame:
    """
    Compare benchmark frequency models on train and test samples.
    """
    rows = []

    for model_name, fitted_model in fitted_models.items():
        for sample_name, sample_df in {"train": train, "test": test}.items():
            expected_claims = compute_expected_claims(fitted_model, sample_df)

            observed_claims = sample_df["claim_count"].sum()
            exposure = sample_df["exposure"].sum()
            total_expected_claims = expected_claims.sum()

            deviance = poisson_deviance(
                observed_counts=sample_df["claim_count"].to_numpy(),
                expected_counts=expected_claims,
            )

            rows.append(
                {
                    "model": model_name,
                    "sample": sample_name,
                    "records": len(sample_df),
                    "exposure": exposure,
                    "observed_claims": observed_claims,
                    "expected_claims": total_expected_claims,
                    "observed_frequency": observed_claims / exposure,
                    "expected_frequency": total_expected_claims / exposure,
                    "poisson_deviance": deviance,
                    "mean_poisson_deviance": deviance / len(sample_df),
                    "train_aic": fitted_model.aic,
                }
            )

    return pd.DataFrame(rows)


def create_coefficient_table(result) -> pd.DataFrame:
    """
    Create coefficient and relativity table for the benchmark GLM.
    """
    conf_int = result.conf_int()

    table = pd.DataFrame(
        {
            "term": result.params.index,
            "coefficient": result.params.values,
            "std_error": result.bse.values,
            "z_value": result.tvalues.values,
            "p_value": result.pvalues.values,
            "ci_lower": conf_int[0].values,
            "ci_upper": conf_int[1].values,
        }
    )

    table["relativity"] = np.exp(table["coefficient"])
    table["relativity_ci_lower"] = np.exp(table["ci_lower"])
    table["relativity_ci_upper"] = np.exp(table["ci_upper"])

    return table


def score_policy_data(
    df: pd.DataFrame,
    fitted_benchmark_model,
) -> pd.DataFrame:
    """
    Add benchmark GLM expected claim counts and expected frequencies.
    """
    scored = df.copy()
    scored["expected_claims_benchmark"] = compute_expected_claims(
        fitted_benchmark_model,
        scored,
    )
    scored["expected_frequency_benchmark"] = (
        scored["expected_claims_benchmark"] / scored["exposure"]
    )

    return scored


def create_observed_expected_table(
    scored: pd.DataFrame,
    segment_columns: list[str],
    min_exposure: float = 100.0,
    min_claims: int = 20,
) -> pd.DataFrame:
    """
    Create observed vs. expected claim diagnostics by segment.
    """
    summary = (
        scored.groupby(segment_columns, observed=True)
        .agg(
            records=("policy_id", "count"),
            policies=("policy_id", "nunique"),
            exposure=("exposure", "sum"),
            observed_claims=("claim_count", "sum"),
            expected_claims=("expected_claims_benchmark", "sum"),
        )
        .reset_index()
    )

    summary["observed_frequency"] = summary["observed_claims"] / summary["exposure"]
    summary["expected_frequency"] = summary["expected_claims"] / summary["exposure"]
    summary["oe_ratio"] = summary["observed_claims"] / summary["expected_claims"]
    summary["claim_difference"] = (
        summary["observed_claims"] - summary["expected_claims"]
    )

    summary["credibility_flag"] = np.select(
        [
            summary["exposure"] < min_exposure,
            summary["observed_claims"] < min_claims,
        ],
        [
            "Low exposure",
            "Low claim count",
        ],
        default="Reviewable",
    )

    summary["diagnostic_note"] = np.select(
        [
            (summary["oe_ratio"] >= 1.20)
            & (summary["credibility_flag"] == "Reviewable"),
            (summary["oe_ratio"] >= 1.20)
            & (summary["credibility_flag"] != "Reviewable"),
            (summary["oe_ratio"] <= 0.80)
            & (summary["credibility_flag"] == "Reviewable"),
        ],
        [
            "Observed claims exceed model expectation",
            "High O/E but limited credibility",
            "Observed claims below model expectation",
        ],
        default="Monitor",
    )

    summary["suggested_review"] = np.select(
        [
            summary["diagnostic_note"] == "Observed claims exceed model expectation",
            summary["diagnostic_note"] == "High O/E but limited credibility",
            summary["diagnostic_note"] == "Observed claims below model expectation",
        ],
        [
            "Review pricing relativities, underwriting rules, and possible missing interactions",
            "Avoid direct repricing; monitor or aggregate with related segments",
            "Check whether the model is overestimating risk or the segment is favorably selected",
        ],
        default="No immediate action; include in recurring monitoring",
    )

    return summary.sort_values(
        ["oe_ratio", "observed_claims"],
        ascending=[False, False],
    )


def round_numeric_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round numeric columns for cleaner exported tables.
    """
    output = df.copy()

    integer_columns = {
        "records",
        "policies",
        "observed_claims",
        "parameters",
        "rank",
    }

    two_decimal_columns = {
        "exposure",
        "expected_claims",
        "claim_difference",
    }

    six_decimal_columns = {
        "observed_frequency",
        "expected_frequency",
        "oe_ratio",
        "total_oe_ratio",
        "deviance_lift_vs_best",
        "coefficient",
        "std_error",
        "z_value",
        "p_value",
        "ci_lower",
        "ci_upper",
        "relativity",
        "relativity_ci_lower",
        "relativity_ci_upper",
        "poisson_deviance",
        "mean_poisson_deviance",
        "train_aic",
    }

    for column in output.select_dtypes(include=[np.number]).columns:
        if column in integer_columns:
            output[column] = output[column].astype(int)
        elif column in two_decimal_columns:
            output[column] = output[column].round(2)
        elif column in six_decimal_columns:
            output[column] = output[column].round(6)
        else:
            output[column] = output[column].round(6)

    return output


def write_frequency_model_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Write frequency model outputs to CSV and Excel.
    """
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_EXCEL.mkdir(parents=True, exist_ok=True)

    for name, table in outputs.items():
        table.to_csv(OUTPUT_TABLES / f"{name}.csv", index=False)

    excel_sheets = {
        "frequency_model_comparison": "model_comparison",
        "frequency_model_coefficients": "coefficients",
        "frequency_oe_by_territory_age": "oe_territory_age",
        "frequency_oe_by_territory_vehicle": "oe_territory_vehicle",
        "frequency_oe_by_vehicle_age": "oe_vehicle_age",
    }

    missing_outputs = set(excel_sheets).difference(outputs)

    if missing_outputs:
        raise KeyError(
            f"Missing required output tables for Excel export: {sorted(missing_outputs)}"
        )

    excel_path = OUTPUT_EXCEL / "frequency_model_review.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for output_name, sheet_name in excel_sheets.items():
            outputs[output_name].to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )


def run_frequency_model() -> None:
    """
    Run the benchmark frequency model layer.
    """
    df = load_policy_data()

    train, test = split_train_test(df)

    null_result = fit_poisson_glm(NULL_FORMULA, train)
    benchmark_result = fit_poisson_glm(MODEL_FORMULA, train)

    fitted_models = {
        "null_offset_only": null_result,
        "benchmark_frequency_glm": benchmark_result,
    }

    model_comparison = create_model_comparison(
        train=train,
        test=test,
        fitted_models=fitted_models,
    )

    coefficient_table = create_coefficient_table(benchmark_result)

    scored = score_policy_data(df, benchmark_result)

    oe_by_territory_age = create_observed_expected_table(
        scored,
        ["territory", "driver_age_band"],
    )

    oe_by_territory_vehicle = create_observed_expected_table(
        scored,
        ["territory", "vehicle_type"],
    )

    oe_by_vehicle_age = create_observed_expected_table(
        scored,
        ["vehicle_type", "vehicle_age_band"],
    )

    outputs = {
        "frequency_model_comparison": round_numeric_outputs(model_comparison),
        "frequency_model_coefficients": round_numeric_outputs(coefficient_table),
        "frequency_oe_by_territory_age": round_numeric_outputs(oe_by_territory_age),
        "frequency_oe_by_territory_vehicle": round_numeric_outputs(
            oe_by_territory_vehicle
        ),
        "frequency_oe_by_vehicle_age": round_numeric_outputs(oe_by_vehicle_age),
    }

    write_frequency_model_outputs(outputs)

    print("Frequency model outputs created:")
    for name in outputs:
        print(f"- outputs/tables/{name}.csv")

    print("- outputs/excel/frequency_model_review.xlsx")


def add_centered_age_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add centered numeric age features using training-sample means.
    """
    required_columns = {
        "driver_age",
        "vehicle_age",
        "driver_age_band",
        "vehicle_age_band",
    }

    missing_train = required_columns.difference(train.columns)
    missing_test = required_columns.difference(test.columns)

    if missing_train:
        raise ValueError(f"Missing required train columns: {sorted(missing_train)}")

    if missing_test:
        raise ValueError(f"Missing required test columns: {sorted(missing_test)}")

    train_output = train.copy()
    test_output = test.copy()

    driver_age_mean = train_output["driver_age"].mean()
    vehicle_age_mean = train_output["vehicle_age"].mean()

    train_output["driver_age_centered"] = (
        train_output["driver_age"] - driver_age_mean
    )
    test_output["driver_age_centered"] = (
        test_output["driver_age"] - driver_age_mean
    )

    train_output["vehicle_age_centered"] = (
        train_output["vehicle_age"] - vehicle_age_mean
    )
    test_output["vehicle_age_centered"] = (
        test_output["vehicle_age"] - vehicle_age_mean
    )

    return train_output, test_output


def fit_model_specifications(
    train: pd.DataFrame,
    model_specs: dict[str, str],
) -> dict[str, object]:
    """
    Fit all candidate Poisson GLM specifications on the training sample.
    """
    fitted_models = {}

    for model_name, formula in model_specs.items():
        fitted_models[model_name] = fit_poisson_glm(formula, train)

    return fitted_models


def create_specification_comparison(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fitted_models: dict[str, object],
) -> pd.DataFrame:
    """
    Compare fitted frequency model specifications on train and test samples.
    """
    rows = []

    samples = {
        "train": train,
        "test": test,
    }

    for model_name, fitted_model in fitted_models.items():
        for sample_name, sample_df in samples.items():
            expected_claims = compute_expected_claims(fitted_model, sample_df)

            observed_claims = sample_df["claim_count"].sum()
            exposure = sample_df["exposure"].sum()
            total_expected_claims = expected_claims.sum()

            deviance = poisson_deviance(
                observed_counts=sample_df["claim_count"].to_numpy(),
                expected_counts=expected_claims,
            )

            rows.append(
                {
                    "model": model_name,
                    "sample": sample_name,
                    "records": len(sample_df),
                    "parameters": int(fitted_model.df_model + 1),
                    "exposure": exposure,
                    "observed_claims": observed_claims,
                    "expected_claims": total_expected_claims,
                    "observed_frequency": observed_claims / exposure,
                    "expected_frequency": total_expected_claims / exposure,
                    "total_oe_ratio": observed_claims / total_expected_claims,
                    "poisson_deviance": deviance,
                    "mean_poisson_deviance": deviance / len(sample_df),
                    "train_aic": fitted_model.aic,
                }
            )

    return pd.DataFrame(rows)


def create_test_ranking(model_comparison: pd.DataFrame) -> pd.DataFrame:
    """
    Rank model specifications by test mean Poisson deviance.
    """
    test_results = model_comparison.query("sample == 'test'").copy()

    test_results = test_results.sort_values(
        ["mean_poisson_deviance", "parameters"],
        ascending=[True, True],
    )

    test_results["rank"] = np.arange(1, len(test_results) + 1)

    best_deviance = test_results["mean_poisson_deviance"].iloc[0]

    test_results["deviance_lift_vs_best"] = (
        test_results["mean_poisson_deviance"] / best_deviance - 1
    )

    return test_results[
        [
            "rank",
            "model",
            "parameters",
            "exposure",
            "observed_claims",
            "expected_claims",
            "total_oe_ratio",
            "mean_poisson_deviance",
            "deviance_lift_vs_best",
            "train_aic",
        ]
    ]


def create_modeling_rationale() -> pd.DataFrame:
    """
    Explain why each candidate specification is included.
    """
    return pd.DataFrame(
        [
            {
                "model": "null_offset_only",
                "purpose": "Baseline model with exposure only",
                "interpretation": (
                    "Tests whether any feature-based specification improves over "
                    "using a single portfolio-level frequency."
                ),
            },
            {
                "model": "categorical_age_bands",
                "purpose": "Interpretable benchmark model",
                "interpretation": (
                    "Uses categorical age bands and treatment coding to produce "
                    "relativities that are easy to explain in a pricing review."
                ),
            },
            {
                "model": "continuous_linear_age",
                "purpose": "Simple continuous-age alternative",
                "interpretation": (
                    "Tests whether raw driver age and vehicle age can be represented "
                    "with a log-linear relationship."
                ),
            },
            {
                "model": "continuous_quadratic_age",
                "purpose": "Nonlinear continuous-age alternative",
                "interpretation": (
                    "Allows curvature in driver age and vehicle age while keeping "
                    "the specification compact."
                ),
            },
            {
                "model": "hybrid_driver_band_vehicle_quadratic",
                "purpose": "Hybrid actuarial specification",
                "interpretation": (
                    "Keeps driver age bands for interpretability and allows a smooth "
                    "nonlinear effect for vehicle age."
                ),
            },
        ]
    )


def write_specification_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Write model specification comparison outputs to CSV and Excel.
    """
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_EXCEL.mkdir(parents=True, exist_ok=True)

    for name, table in outputs.items():
        table.to_csv(OUTPUT_TABLES / f"{name}.csv", index=False)

    excel_sheets = {
        "frequency_model_specification_comparison": "comparison",
        "frequency_model_specification_ranking": "test_ranking",
        "frequency_model_specification_rationale": "rationale",
    }

    missing_outputs = set(excel_sheets).difference(outputs)

    if missing_outputs:
        raise KeyError(
            f"Missing required output tables for Excel export: {sorted(missing_outputs)}"
        )

    with pd.ExcelWriter(SPECIFICATION_EXCEL_PATH, engine="openpyxl") as writer:
        for output_name, sheet_name in excel_sheets.items():
            outputs[output_name].to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )


def run_model_specification_comparison() -> None:
    """
    Run the model specification comparison layer.
    """
    df = load_policy_data()

    train, test = split_train_test(df)
    train, test = add_centered_age_features(train, test)

    fitted_models = fit_model_specifications(
        train=train,
        model_specs=MODEL_SPECS,
    )

    model_comparison = create_specification_comparison(
        train=train,
        test=test,
        fitted_models=fitted_models,
    )

    test_ranking = create_test_ranking(model_comparison)
    modeling_rationale = create_modeling_rationale()

    outputs = {
        "frequency_model_specification_comparison": round_numeric_outputs(
            model_comparison
        ),
        "frequency_model_specification_ranking": round_numeric_outputs(test_ranking),
        "frequency_model_specification_rationale": modeling_rationale,
    }

    write_specification_outputs(outputs)

    print("Frequency model specification comparison outputs created:")
    for name in outputs:
        print(f"- outputs/tables/{name}.csv")

    print("- outputs/excel/frequency_model_specification_comparison.xlsx")


def score_policy_data_for_diagnostics(
    df: pd.DataFrame,
    fitted_model,
    sample_name: str,
) -> pd.DataFrame:
    """
    Add benchmark model expected claim counts, expected frequencies, and residuals.
    """
    scored = df.copy()

    observed_counts = scored["claim_count"].to_numpy(dtype=float)
    expected_counts = compute_expected_claims(fitted_model, scored)

    scored["sample"] = sample_name
    scored["expected_claims_benchmark"] = expected_counts
    scored["expected_frequency_benchmark"] = (
        scored["expected_claims_benchmark"] / scored["exposure"]
    )

    scored["pearson_residual"] = (
        observed_counts - expected_counts
    ) / np.sqrt(expected_counts)

    scored["deviance_residual"] = compute_deviance_residuals(
        observed_counts=observed_counts,
        expected_counts=expected_counts,
    )

    scored["abs_deviance_residual"] = scored["deviance_residual"].abs()

    scored["quantile_residual"] = compute_randomized_quantile_residuals(
        observed_counts=observed_counts,
        expected_counts=expected_counts,
    )

    scored["abs_quantile_residual"] = scored["quantile_residual"].abs()

    return scored


def compute_deviance_residuals(
    observed_counts: np.ndarray,
    expected_counts: np.ndarray,
) -> np.ndarray:
    """
    Compute Poisson deviance residuals.
    """
    observed_counts = np.asarray(observed_counts, dtype=float)
    expected_counts = np.asarray(expected_counts, dtype=float).clip(min=1e-12)

    term = np.zeros_like(observed_counts, dtype=float)

    positive = observed_counts > 0
    term[positive] = observed_counts[positive] * np.log(
        observed_counts[positive] / expected_counts[positive]
    )

    term -= observed_counts - expected_counts

    unit_deviance = np.maximum(2 * term, 0.0)

    residual_sign = np.where(observed_counts >= expected_counts, 1.0, -1.0)

    return residual_sign * np.sqrt(unit_deviance)


def compute_randomized_quantile_residuals(
    observed_counts: np.ndarray,
    expected_counts: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute randomized quantile residuals for Poisson claim counts.
    """
    observed_counts = np.asarray(observed_counts, dtype=int)
    expected_counts = np.asarray(expected_counts, dtype=float).clip(min=1e-12)

    rng = np.random.default_rng(seed)

    lower_probability = poisson.cdf(observed_counts - 1, expected_counts)
    upper_probability = poisson.cdf(observed_counts, expected_counts)

    uniform_probability = rng.uniform(
        lower_probability,
        upper_probability,
    )

    uniform_probability = np.clip(uniform_probability, 1e-12, 1 - 1e-12)

    return norm.ppf(uniform_probability)


def create_diagnostics_summary(
    scored_samples: dict[str, pd.DataFrame],
    fitted_model,
) -> pd.DataFrame:
    """
    Create model-level diagnostic metrics by sample.
    """
    rows = []

    n_parameters = int(fitted_model.df_model + 1)

    for sample_name, sample_df in scored_samples.items():
        observed_counts = sample_df["claim_count"].to_numpy(dtype=float)
        expected_counts = sample_df["expected_claims_benchmark"].to_numpy(dtype=float)

        exposure = sample_df["exposure"].sum()
        observed_claims = observed_counts.sum()
        expected_claims = expected_counts.sum()

        deviance = poisson_deviance(
            observed_counts=observed_counts,
            expected_counts=expected_counts,
        )

        pearson_chi_square = np.sum(
            ((observed_counts - expected_counts) ** 2) / expected_counts
        )

        dispersion_df = max(len(sample_df) - n_parameters, 1)

        rows.append(
            {
                "sample": sample_name,
                "records": len(sample_df),
                "n_parameters": n_parameters,
                "dispersion_df": dispersion_df,
                "exposure": exposure,
                "observed_claims": observed_claims,
                "expected_claims": expected_claims,
                "observed_frequency": observed_claims / exposure,
                "expected_frequency": expected_claims / exposure,
                "total_oe_ratio": observed_claims / expected_claims,
                "poisson_deviance": deviance,
                "mean_poisson_deviance": deviance / len(sample_df),
                "pearson_chi_square": pearson_chi_square,
                "pearson_dispersion": pearson_chi_square / dispersion_df,
                "deviance_dispersion": deviance / dispersion_df,
                "mean_pearson_residual": sample_df["pearson_residual"].mean(),
                "std_pearson_residual": sample_df["pearson_residual"].std(),
                "mean_deviance_residual": sample_df["deviance_residual"].mean(),
                "std_deviance_residual": sample_df["deviance_residual"].std(),
                "mean_quantile_residual": sample_df["quantile_residual"].mean(),
                "std_quantile_residual": sample_df["quantile_residual"].std(),
                "q025_quantile_residual": sample_df["quantile_residual"].quantile(
                    0.025
                ),
                "median_quantile_residual": sample_df[
                    "quantile_residual"
                ].median(),
                "q975_quantile_residual": sample_df["quantile_residual"].quantile(
                    0.975
                ),
            }
        )

    return pd.DataFrame(rows)


def create_calibration_by_decile(
    scored: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Create calibration diagnostics by expected frequency decile.
    """
    output = scored.copy()

    risk_rank = output["expected_frequency_benchmark"].rank(method="first")

    output["risk_decile"] = pd.qcut(
        risk_rank,
        q=n_bins,
        labels=False,
    ) + 1

    summary = (
        output.groupby("risk_decile", observed=True)
        .agg(
            records=("policy_id", "count"),
            policies=("policy_id", "nunique"),
            exposure=("exposure", "sum"),
            observed_claims=("claim_count", "sum"),
            expected_claims=("expected_claims_benchmark", "sum"),
            avg_expected_frequency=("expected_frequency_benchmark", "mean"),
            mean_deviance_residual=("deviance_residual", "mean"),
            std_deviance_residual=("deviance_residual", "std"),
        )
        .reset_index()
    )

    summary["observed_frequency"] = (
        summary["observed_claims"] / summary["exposure"]
    )
    summary["expected_frequency"] = (
        summary["expected_claims"] / summary["exposure"]
    )
    summary["oe_ratio"] = summary["observed_claims"] / summary["expected_claims"]
    summary["claim_difference"] = (
        summary["observed_claims"] - summary["expected_claims"]
    )

    return summary


def create_top_residuals(
    scored: pd.DataFrame,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Return policy-period records with the largest absolute quantile residuals.
    """
    columns = [
        "sample",
        "policy_id",
        "exposure",
        "claim_count",
        "expected_claims_benchmark",
        "expected_frequency_benchmark",
        "pearson_residual",
        "deviance_residual",
        "abs_deviance_residual",
        "quantile_residual",
        "abs_quantile_residual",
        "territory",
        "driver_age",
        "driver_age_band",
        "vehicle_age",
        "vehicle_age_band",
        "vehicle_type",
    ]

    available_columns = [column for column in columns if column in scored.columns]

    return (
        scored.sort_values("abs_quantile_residual", ascending=False)
        .head(top_n)[available_columns]
        .copy()
    )


def create_diagnostics_rationale() -> pd.DataFrame:
    """
    Document why each diagnostic is included.
    """
    return pd.DataFrame(
        [
            {
                "diagnostic": "calibration_by_decile",
                "purpose": (
                    "Compare observed and expected claims across risk-ranked groups."
                ),
                "interpretation": (
                    "A well-calibrated benchmark should show observed and expected "
                    "frequencies that are reasonably close across deciles, with no "
                    "persistent O/E pattern by expected-risk band."
                ),
            },
            {
                "diagnostic": "pearson_dispersion",
                "purpose": (
                    "Check whether claim count variability exceeds the Poisson "
                    "variance assumption."
                ),
                "interpretation": (
                    "Values materially above 1 suggest overdispersion relative to "
                    "the Poisson assumption. This may motivate quasi-Poisson, "
                    "negative binomial, omitted-variable review, or richer segmentation."
                ),
            },
            {
                "diagnostic": "deviance_dispersion",
                "purpose": (
                    "Summarize model lack of fit using deviance relative to residual "
                    "degrees of freedom."
                ),
                "interpretation": (
                    "Values materially above 1 indicate that the fitted Poisson model "
                    "does not explain the observed deviance as well as expected under "
                    "the benchmark assumptions."
                ),
            },
            {
                "diagnostic": "randomized_quantile_residuals",
                "purpose": (
                    "Assess residual distribution and systematic model structure for "
                    "discrete claim counts."
                ),
                "interpretation": (
                    "For a correctly specified discrete response model, randomized "
                    "quantile residuals should be approximately standard normal."
                ),
            },
            {
                "diagnostic": "train_test_diagnostics",
                "purpose": (
                    "Compare model behavior in training and held-out samples."
                ),
                "interpretation": (
                    "Similar train and test diagnostics support model stability. Large "
                    "train-test gaps may indicate overfitting, sample instability, or "
                    "weak generalization."
                ),
            },
        ]
    )


def round_diagnostic_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round diagnostic outputs for cleaner exported tables.
    """
    output = df.copy()

    integer_columns = {
        "records",
        "policies",
        "observed_claims",
        "n_parameters",
        "dispersion_df",
        "risk_decile",
        "claim_count",
        "policy_id",
    }

    two_decimal_columns = {
        "exposure",
        "expected_claims",
        "expected_claims_benchmark",
        "claim_difference",
    }

    six_decimal_columns = {
        "observed_frequency",
        "expected_frequency",
        "avg_expected_frequency",
        "expected_frequency_benchmark",
        "total_oe_ratio",
        "oe_ratio",
        "poisson_deviance",
        "mean_poisson_deviance",
        "pearson_chi_square",
        "pearson_dispersion",
        "deviance_dispersion",
        "pearson_residual",
        "deviance_residual",
        "abs_deviance_residual",
        "mean_pearson_residual",
        "std_pearson_residual",
        "mean_deviance_residual",
        "std_deviance_residual",
        "mean_quantile_residual",
        "std_quantile_residual",
        "q025_quantile_residual",
        "median_quantile_residual",
        "q975_quantile_residual",
    }

    for column in output.select_dtypes(include=[np.number]).columns:
        if column in integer_columns:
            output[column] = output[column].astype(int)
        elif column in two_decimal_columns:
            output[column] = output[column].round(2)
        elif column in six_decimal_columns:
            output[column] = output[column].round(6)
        else:
            output[column] = output[column].round(6)

    return output


def save_calibration_chart(calibration: pd.DataFrame) -> None:
    """
    Save observed vs expected frequency by risk decile.
    """
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        calibration["risk_decile"],
        calibration["observed_frequency"],
        marker="o",
        label="Observed frequency",
    )
    ax.plot(
        calibration["risk_decile"],
        calibration["expected_frequency"],
        marker="o",
        label="Expected frequency",
    )

    ax.set_title("Observed vs Expected Frequency by Risk Decile")
    ax.set_xlabel("Risk decile")
    ax.set_ylabel("Claim frequency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        OUTPUT_FIGURES / "frequency_model_calibration_by_decile.png",
        dpi=150,
    )
    plt.close(fig)


def save_oe_chart(calibration: pd.DataFrame) -> None:
    """
    Save O/E ratio by risk decile.
    """
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        calibration["risk_decile"].astype(str),
        calibration["oe_ratio"],
    )
    ax.axhline(1.0, linewidth=1)

    ax.set_title("Observed-to-Expected Ratio by Risk Decile")
    ax.set_xlabel("Risk decile")
    ax.set_ylabel("O/E ratio")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    fig.savefig(
        OUTPUT_FIGURES / "frequency_model_oe_by_decile.png",
        dpi=150,
    )
    plt.close(fig)


def save_quantile_residual_chart(scored: pd.DataFrame) -> None:
    """
    Save randomized quantile residuals against expected frequency.
    """
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        scored["expected_frequency_benchmark"],
        scored["quantile_residual"],
        alpha=0.30,
        s=12,
    )
    ax.axhline(0.0, linewidth=1)

    ax.set_title("Randomized Quantile Residuals vs Expected Frequency")
    ax.set_xlabel("Expected frequency")
    ax.set_ylabel("Randomized quantile residual")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(
        OUTPUT_FIGURES / "frequency_model_quantile_residuals.png",
        dpi=150,
    )
    plt.close(fig)


def save_quantile_residual_qq_chart(scored: pd.DataFrame) -> None:
    """
    Save QQ plot for randomized quantile residuals.
    """
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

    residuals = np.sort(scored["quantile_residual"].dropna().to_numpy())
    n = len(residuals)

    theoretical_quantiles = norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(theoretical_quantiles, residuals, alpha=0.35, s=12)

    lower = min(theoretical_quantiles.min(), residuals.min())
    upper = max(theoretical_quantiles.max(), residuals.max())
    ax.plot([lower, upper], [lower, upper], linewidth=1)

    ax.set_title("QQ Plot of Randomized Quantile Residuals")
    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_ylabel("Observed quantile residuals")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(
        OUTPUT_FIGURES / "frequency_model_quantile_residual_qq.png",
        dpi=150,
    )
    plt.close(fig)


def write_diagnostic_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Write model diagnostic outputs to CSV and Excel.
    """
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_EXCEL.mkdir(parents=True, exist_ok=True)

    for name, table in outputs.items():
        table.to_csv(OUTPUT_TABLES / f"{name}.csv", index=False)

    excel_sheets = {
        "frequency_model_diagnostics_summary": "summary",
        "frequency_model_calibration_by_decile": "calibration_decile",
        "frequency_model_top_residuals": "top_residuals",
        "frequency_model_diagnostics_rationale": "rationale",
    }

    missing_outputs = set(excel_sheets).difference(outputs)

    if missing_outputs:
        raise KeyError(
            f"Missing required output tables for Excel export: {sorted(missing_outputs)}"
        )

    with pd.ExcelWriter(DIAGNOSTICS_EXCEL_PATH, engine="openpyxl") as writer:
        for output_name, sheet_name in excel_sheets.items():
            outputs[output_name].to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )


def run_frequency_model_diagnostics() -> None:
    """
    Run the frequency model diagnostics layer.
    """
    df = load_policy_data()

    train, test = split_train_test(df)

    benchmark_model = fit_poisson_glm(MODEL_FORMULA, train)

    scored_train = score_policy_data_for_diagnostics(
        train,
        fitted_model=benchmark_model,
        sample_name="train",
    )

    scored_test = score_policy_data_for_diagnostics(
        test,
        fitted_model=benchmark_model,
        sample_name="test",
    )

    diagnostics_summary = create_diagnostics_summary(
        scored_samples={
            "train": scored_train,
            "test": scored_test,
        },
        fitted_model=benchmark_model,
    )

    calibration_by_decile = create_calibration_by_decile(scored_test)

    scored_all = pd.concat([scored_train, scored_test], ignore_index=True)

    top_residuals = create_top_residuals(scored_all)

    diagnostics_rationale = create_diagnostics_rationale()

    outputs = {
        "frequency_model_diagnostics_summary": round_diagnostic_outputs(
            diagnostics_summary
        ),
        "frequency_model_calibration_by_decile": round_diagnostic_outputs(
            calibration_by_decile
        ),
        "frequency_model_top_residuals": round_diagnostic_outputs(top_residuals),
        "frequency_model_diagnostics_rationale": diagnostics_rationale,
    }

    write_diagnostic_outputs(outputs)

    save_calibration_chart(calibration_by_decile)
    save_oe_chart(calibration_by_decile)
    save_quantile_residual_chart(scored_test)
    save_quantile_residual_qq_chart(scored_test)

    print("Frequency model diagnostics created:")
    for name in outputs:
        print(f"- outputs/tables/{name}.csv")

    print("- outputs/excel/frequency_model_diagnostics.xlsx")
    print("- outputs/figures/frequency_model_calibration_by_decile.png")
    print("- outputs/figures/frequency_model_oe_by_decile.png")
    print("- outputs/figures/frequency_model_quantile_residuals.png")
    print("- outputs/figures/frequency_model_quantile_residual_qq.png")


def main() -> None:
    """
    Run the full frequency analysis workflow.
    """
    run_frequency_model()
    run_model_specification_comparison()
    run_frequency_model_diagnostics()


if __name__ == "__main__":
    main()