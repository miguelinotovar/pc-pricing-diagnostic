import numpy as np
import pandas as pd

from pc_pricing_diagnostic.frequency_model import (
    MODEL_FORMULA as CATEGORICAL_FORMULA,
    NULL_FORMULA,
    OUTPUT_EXCEL,
    OUTPUT_TABLES,
    compute_expected_claims,
    fit_poisson_glm,
    load_policy_data,
    poisson_deviance,
    round_numeric_outputs,
    split_train_test,
)


EXCEL_PATH = OUTPUT_EXCEL / "frequency_model_specification_comparison.xlsx"


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


def add_centered_age_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add centered numeric age features using training-sample means.

    The centering constants are estimated on the training sample and then applied
    to both train and test. This avoids using test-sample information in feature
    preparation.
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

    Test mean Poisson deviance is the main out-of-sample comparison metric.
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

    Lower test mean deviance indicates better out-of-sample performance under
    Poisson deviance loss.
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


def write_outputs(outputs: dict[str, pd.DataFrame]) -> None:
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

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        for output_name, sheet_name in excel_sheets.items():
            outputs[output_name].to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )


def main() -> None:
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

    write_outputs(outputs)

    print("Frequency model specification comparison outputs created:")
    for name in outputs:
        print(f"- outputs/tables/{name}.csv")

    print("- outputs/excel/frequency_model_specification_comparison.xlsx")


if __name__ == "__main__":
    main()