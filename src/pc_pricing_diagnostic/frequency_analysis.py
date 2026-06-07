from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

from pc_pricing_diagnostic.config import OUTPUT_EXCEL, OUTPUT_TABLES, PROCESSED_DATA


DATA_PATH = PROCESSED_DATA / "synthetic_policy_data.csv"


MODEL_FORMULA = (
    'claim_count ~ '
    'C(territory, Treatment(reference="Rural")) + '
    'C(driver_age_band, Treatment(reference="40-64")) + '
    'C(vehicle_age_band, Treatment(reference="4-7")) + '
    'C(vehicle_type, Treatment(reference="Sedan"))'
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

    The response is the observed claim count for each policy-period. The offset
    enters the linear predictor with a fixed coefficient of 1, so exposure is treated
    as a known pro-rata temporis volume measure rather than as a risk covariate to
    be estimated.

    With a log link, the model is:

        log(E[N_i]) = log(exposure_i) + x_i' beta

    equivalently:

        E[N_i] = exposure_i * exp(x_i' beta)

    Therefore, exp(x_i' beta) is interpreted as the expected claim frequency per
    unit of exposure, while E[N_i] remains the expected claim count for the observed
    policy-period.
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

    The fitted GLM returns the conditional mean mu_hat_i for each record:

        mu_hat_i = exposure_i * exp(x_i' beta_hat)
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

    observed_counts contains the observed claim counts N_i.
    expected_counts contains the fitted conditional means mu_hat_i.

    The function computes:

        2 * sum_i [N_i log(N_i / mu_hat_i) - N_i + mu_hat_i]

    with the convention that N_i log(N_i / mu_hat_i) is zero when N_i = 0.
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
    fitted_benchmark_model,
) -> pd.DataFrame:
    """
    Add benchmark GLM expected claim counts and expected frequencies to each
    policy-period record.

    expected_claims_benchmark is the fitted conditional mean for the claim count.
    expected_frequency_benchmark is that expected count divided by exposure.
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

    CSV exports are written for every table in outputs. The Excel workbook uses
    an explicit sheet map so the client-facing workbook has stable sheet names.
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