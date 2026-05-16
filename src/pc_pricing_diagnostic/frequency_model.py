from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "synthetic_policy_data.csv"
OUTPUT_TABLES = ROOT / "outputs" / "tables"
OUTPUT_EXCEL = ROOT / "outputs" / "excel"


MODEL_FORMULA = (
    "claim_count ~ C(territory) + C(driver_age_band) "
    "+ C(vehicle_age_band) + C(vehicle_type)"
)

NULL_FORMULA = "claim_count ~ 1"


def load_policy_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the policy-period dataset used for frequency modeling.

    The model uses claim_count as the response and exposure as an offset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Policy data not found at {path}. "
            "Run synthetic_data.py before running the frequency model."
        )

    df = pd.read_csv(path)

    required_columns = {
        "policy_id",
        "exposure",
        "claim_count",
        "territory",
        "driver_age_band",
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

    The split is stratified by whether a policy-period has at least one claim, so
    both samples keep a similar claim incidence structure.
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

    The offset means the model estimates annual claim frequency while the response
    remains the observed claim count for the policy-period.
    """
    offset = np.log(df["exposure"])

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Poisson(),
        offset=offset,
    )

    return model.fit()


def predict_expected_claims(result, df: pd.DataFrame) -> np.ndarray:
    """
    Predict expected claim counts for each policy-period.
    """
    offset = np.log(df["exposure"])
    predictions = result.predict(df, offset=offset)

    return np.asarray(predictions).clip(min=1e-12)


def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Poisson deviance.

    Lower deviance indicates better fit. This is the natural diagnostic scale for
    Poisson frequency models.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float).clip(min=1e-12)

    term = np.zeros_like(y_true, dtype=float)

    positive = y_true > 0
    term[positive] = y_true[positive] * np.log(y_true[positive] / y_pred[positive])
    term -= y_true - y_pred

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

    for model_name, result in fitted_models.items():
        for sample_name, sample_df in {"train": train, "test": test}.items():
            expected_claims = predict_expected_claims(result, sample_df)

            observed_claims = sample_df["claim_count"].sum()
            exposure = sample_df["exposure"].sum()
            predicted_claims = expected_claims.sum()

            deviance = poisson_deviance(
                sample_df["claim_count"].to_numpy(),
                expected_claims,
            )

            rows.append(
                {
                    "model": model_name,
                    "sample": sample_name,
                    "records": len(sample_df),
                    "exposure": exposure,
                    "observed_claims": observed_claims,
                    "predicted_claims": predicted_claims,
                    "observed_frequency": observed_claims / exposure,
                    "predicted_frequency": predicted_claims / exposure,
                    "poisson_deviance": deviance,
                    "mean_poisson_deviance": deviance / len(sample_df),
                    "train_aic": result.aic,
                }
            )

    return pd.DataFrame(rows)


def create_coefficient_table(result) -> pd.DataFrame:
    """
    Create coefficient and relativity table for the benchmark GLM.

    Exponentiated coefficients are shown as relativities. For categorical variables,
    they are interpreted relative to the model's reference category.
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
    benchmark_result,
) -> pd.DataFrame:
    """
    Add model-predicted expected claims to policy-period data.
    """
    scored = df.copy()
    scored["expected_claims_benchmark"] = predict_expected_claims(
        benchmark_result,
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

    O/E above 1 means observed claims exceed model-expected claims.
    O/E below 1 means observed claims are lower than model-expected claims.
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
        "coefficient",
        "std_error",
        "z_value",
        "p_value",
        "ci_lower",
        "ci_upper",
        "relativity",
        "relativity_ci_lower",
        "relativity_ci_upper",
        "predicted_claims",
        "predicted_frequency",
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


def write_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Write frequency model outputs to CSV and Excel.
    """
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_EXCEL.mkdir(parents=True, exist_ok=True)

    for name, table in outputs.items():
        table.to_csv(OUTPUT_TABLES / f"{name}.csv", index=False)

    excel_path = OUTPUT_EXCEL / "frequency_model_review.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        outputs["frequency_model_comparison"].to_excel(
            writer,
            sheet_name="model_comparison",
            index=False,
        )
        outputs["frequency_model_coefficients"].to_excel(
            writer,
            sheet_name="coefficients",
            index=False,
        )
        outputs["frequency_oe_by_territory_age"].to_excel(
            writer,
            sheet_name="oe_territory_age",
            index=False,
        )
        outputs["frequency_oe_by_territory_vehicle"].to_excel(
            writer,
            sheet_name="oe_territory_vehicle",
            index=False,
        )
        outputs["frequency_oe_by_vehicle_age"].to_excel(
            writer,
            sheet_name="oe_vehicle_age",
            index=False,
        )


def main() -> None:
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

    write_outputs(outputs)

    print("Frequency model outputs created:")
    for name in outputs:
        print(f"- outputs/tables/{name}.csv")

    print("- outputs/excel/frequency_model_review.xlsx")


if __name__ == "__main__":
    main()