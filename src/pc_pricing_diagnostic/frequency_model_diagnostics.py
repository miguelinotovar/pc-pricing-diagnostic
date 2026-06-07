import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from pc_pricing_diagnostic.config import OUTPUT_EXCEL, OUTPUT_FIGURES, OUTPUT_TABLES
from pc_pricing_diagnostic.frequency_analysis import (
    MODEL_FORMULA,
    compute_expected_claims,
    fit_poisson_glm,
    load_policy_data,
    poisson_deviance,
    split_train_test,
)


EXCEL_PATH = OUTPUT_EXCEL / "frequency_model_diagnostics.xlsx"


def score_policy_data(
    df: pd.DataFrame,
    fitted_model,
    sample_name: str,
) -> pd.DataFrame:
    """
    Add benchmark model expected claim counts, expected frequencies, and residuals
    to a policy-period dataset.
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

    The squared deviance residual contribution equals the individual Poisson
    deviance contribution. The sign indicates whether observed counts are above
    or below expected counts.
    """
    observed_counts = np.asarray(observed_counts, dtype=float)
    expected_counts = np.asarray(expected_counts, dtype=float).clip(min=1e-12)

    term = np.zeros_like(observed_counts, dtype=float)

    positive = observed_counts > 0
    term[positive] = observed_counts[positive] * np.log(
        observed_counts[positive] / expected_counts[positive]
    )

    term -= observed_counts - expected_counts

    unit_deviance = 2 * term

    residual_sign = np.where(observed_counts >= expected_counts, 1.0, -1.0)

    return residual_sign * np.sqrt(unit_deviance)


def compute_randomized_quantile_residuals(
    observed_counts: np.ndarray,
    expected_counts: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute randomized quantile residuals for Poisson claim counts.

    For a discrete response, the fitted CDF has jumps. The randomized quantile
    residual samples uniformly inside the probability interval assigned to the
    observed count and maps that probability to the standard normal scale.

    If the Poisson model is correctly specified, these residuals should be
    approximately standard normal.
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

    Pearson and deviance dispersion ratios above 1 may indicate overdispersion
    relative to the Poisson variance assumption.
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
                "q025_quantile_residual": sample_df["quantile_residual"].quantile(0.025),
                "median_quantile_residual": sample_df["quantile_residual"].median(),
                "q975_quantile_residual": sample_df["quantile_residual"].quantile(0.975),
            }
        )

    return pd.DataFrame(rows)


def create_calibration_by_decile(
    scored: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Create calibration diagnostics by expected frequency decile.

    The table checks whether higher expected-risk groups also show higher observed
    frequency and whether observed claims are close to expected claims by risk band.
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
    summary["oe_ratio"] = (
        summary["observed_claims"] / summary["expected_claims"]
    )
    summary["claim_difference"] = (
        summary["observed_claims"] - summary["expected_claims"]
    )

    return summary


def create_top_residuals(
    scored: pd.DataFrame,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Return policy-period records with the largest absolute deviance residuals.

    These are not automatically errors. They are records that contribute strongly
    to model deviance and may deserve data quality or underwriting review.
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
                    "the Poisson assumption Var(N_i | x_i) = E(N_i | x_i). This may "
                    "motivate quasi-Poisson, negative binomial, omitted-variable "
                    "review, or richer segmentation."
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
                    "the benchmark assumptions. This is complementary to Pearson "
                    "dispersion."
                ),
            },
            {
                "diagnostic": "deviance_residuals",
                "purpose": (
                    "Measure signed observation-level contributions to total Poisson "
                    "deviance."
                ),
                "interpretation": (
                    "Large absolute deviance residuals identify records that contribute "
                    "strongly to model deviance. Positive values mean observed claims "
                    "exceed expected claims; negative values mean observed claims are "
                    "below expected claims. For count data, deviance residual plots may "
                    "show discreteness-related patterns, so quantile residuals are used "
                    "as the main graphical residual diagnostic."
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
                    "quantile residuals should be approximately standard normal. "
                    "They are preferred for residual plots in count models because "
                    "Pearson and deviance residuals may show non-normality or banding "
                    "caused by discreteness."
                ),
            },
            {
                "diagnostic": "quantile_residual_qq_plot",
                "purpose": (
                    "Check whether randomized quantile residuals are close to a "
                    "standard normal distribution."
                ),
                "interpretation": (
                    "Strong departures from the reference line suggest tail issues, "
                    "misspecification, omitted variables, overdispersion, or an "
                    "inadequate count distribution."
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

    For count models, randomized quantile residuals are preferred for residual
    plots because Pearson and deviance residuals may show discreteness-related
    patterns.
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

    If the Poisson model is reasonably specified, randomized quantile residuals
    should be approximately standard normal.
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


def write_outputs(outputs: dict[str, pd.DataFrame]) -> None:
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

    benchmark_model = fit_poisson_glm(MODEL_FORMULA, train)

    scored_train = score_policy_data(
        train,
        fitted_model=benchmark_model,
        sample_name="train",
    )

    scored_test = score_policy_data(
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

    write_outputs(outputs)

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


if __name__ == "__main__":
    main()