import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pc_pricing_diagnostic.config import (
    OUTPUT_EXCEL,
    OUTPUT_TABLES,
    PROCESSED_DATA,
)
from pc_pricing_diagnostic.io_utils import write_csv_outputs
from pc_pricing_diagnostic.plot_style import (
    BAR_BLUE,
    BLUE,
    apply_axis_style,
    save_figure,
)


DATA_PATH = PROCESSED_DATA / "synthetic_policy_data.csv"

DRIVER_AGE_BAND_ORDER = ["16-24", "25-39", "40-64", "65+"]
VEHICLE_AGE_BAND_ORDER = ["0-3", "4-7", "8-12", "13+"]
TERRITORY_ORDER = ["Rural", "Suburban", "Urban B", "Urban A"]
VEHICLE_TYPE_ORDER = ["Sedan", "SUV", "Truck", "Sports"]


def load_policy_data(path=DATA_PATH) -> pd.DataFrame:
    """
    Load the synthetic policy-period dataset.

    Each row represents one policy exposure record with exposure, claim count,
    premium, rating variables, and total claim amount for the period.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Policy data not found at {path}. "
            "Run synthetic_data.py before running experience diagnostics."
        )

    df = pd.read_csv(path)

    required_columns = {
        "policy_id",
        "exposure",
        "claim_count",
        "total_claim_amount",
        "earned_premium",
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
        raise ValueError("Exposure must be strictly positive.")

    return apply_categorical_order(df)


def apply_categorical_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply explicit category order for cleaner diagnostics tables and charts.
    """
    output = df.copy()

    output["driver_age_band"] = pd.Categorical(
        output["driver_age_band"],
        categories=DRIVER_AGE_BAND_ORDER,
        ordered=True,
    )

    output["vehicle_age_band"] = pd.Categorical(
        output["vehicle_age_band"],
        categories=VEHICLE_AGE_BAND_ORDER,
        ordered=True,
    )

    output["territory"] = pd.Categorical(
        output["territory"],
        categories=TERRITORY_ORDER,
        ordered=True,
    )

    output["vehicle_type"] = pd.Categorical(
        output["vehicle_type"],
        categories=VEHICLE_TYPE_ORDER,
        ordered=True,
    )

    return output


def portfolio_frequency(df: pd.DataFrame) -> float:
    """
    Compute portfolio-level observed claim frequency per unit of exposure.
    """
    return df["claim_count"].sum() / df["exposure"].sum()


def portfolio_loss_ratio(df: pd.DataFrame) -> float:
    """
    Compute portfolio-level loss ratio.
    """
    return df["total_claim_amount"].sum() / df["earned_premium"].sum()


def create_portfolio_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a one-table portfolio overview.
    """
    total_exposure = df["exposure"].sum()
    total_claims = df["claim_count"].sum()
    total_claim_amount = df["total_claim_amount"].sum()
    total_earned_premium = df["earned_premium"].sum()

    observed_frequency = total_claims / total_exposure
    loss_ratio = total_claim_amount / total_earned_premium
    pure_premium = total_claim_amount / total_exposure
    premium_per_exposure = total_earned_premium / total_exposure

    average_claim_size = (
        total_claim_amount / total_claims if total_claims > 0 else np.nan
    )

    return pd.DataFrame(
        {
            "metric": [
                "records",
                "distinct_policies",
                "total_exposure",
                "total_claims",
                "observed_frequency",
                "total_claim_amount",
                "total_earned_premium",
                "loss_ratio",
                "pure_premium",
                "premium_per_exposure",
                "average_claim_size",
            ],
            "value": [
                len(df),
                df["policy_id"].nunique(),
                total_exposure,
                total_claims,
                observed_frequency,
                total_claim_amount,
                total_earned_premium,
                loss_ratio,
                pure_premium,
                premium_per_exposure,
                average_claim_size,
            ],
        }
    )


def create_segment_experience(
    df: pd.DataFrame,
    segment_columns: list[str],
    min_exposure: float = 100.0,
    min_claims: int = 20,
) -> pd.DataFrame:
    """
    Create segment-level experience.

    The table supports pricing, underwriting, data quality, and monitoring review.
    """
    total_frequency = portfolio_frequency(df)
    total_loss_ratio = portfolio_loss_ratio(df)

    summary = (
        df.groupby(segment_columns, observed=True)
        .agg(
            records=("policy_id", "count"),
            policies=("policy_id", "nunique"),
            exposure=("exposure", "sum"),
            claims=("claim_count", "sum"),
            total_claim_amount=("total_claim_amount", "sum"),
            earned_premium=("earned_premium", "sum"),
        )
        .reset_index()
    )

    summary["observed_frequency"] = summary["claims"] / summary["exposure"]

    summary["average_claim_size"] = np.where(
        summary["claims"] > 0,
        summary["total_claim_amount"] / summary["claims"],
        np.nan,
    )

    summary["pure_premium"] = summary["total_claim_amount"] / summary["exposure"]
    summary["premium_per_exposure"] = summary["earned_premium"] / summary["exposure"]
    summary["loss_ratio"] = summary["total_claim_amount"] / summary["earned_premium"]

    summary["frequency_index"] = summary["observed_frequency"] / total_frequency
    summary["loss_ratio_index"] = summary["loss_ratio"] / total_loss_ratio

    summary["credibility_flag"] = np.select(
        [
            summary["exposure"] < min_exposure,
            summary["claims"] < min_claims,
        ],
        [
            "Low exposure",
            "Low claim count",
        ],
        default="Reviewable",
    )

    return summary.sort_values(
        ["frequency_index", "claims"],
        ascending=[False, False],
    )


def create_frequency_table(
    df: pd.DataFrame,
    group_columns: list[str],
    min_exposure: float = 100.0,
    min_claims: int = 20,
) -> pd.DataFrame:
    """
    Create exploratory frequency and loss-ratio diagnostics by group.
    """
    return create_segment_experience(
        df=df,
        segment_columns=group_columns,
        min_exposure=min_exposure,
        min_claims=min_claims,
    )


def create_high_frequency_segments(
    segment_experience: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Identify segments with high observed frequency relative to the portfolio average.
    """
    high_frequency = segment_experience.copy()

    high_frequency["diagnostic_note"] = np.select(
        [
            (high_frequency["frequency_index"] >= 1.50)
            & (high_frequency["credibility_flag"] == "Reviewable"),
            (high_frequency["frequency_index"] >= 1.50)
            & (high_frequency["credibility_flag"] != "Reviewable"),
            high_frequency["loss_ratio_index"] >= 1.50,
        ],
        [
            "High frequency with sufficient volume",
            "High frequency but limited credibility",
            "High loss ratio relative to portfolio",
        ],
        default="Monitor",
    )

    high_frequency["suggested_review"] = np.select(
        [
            high_frequency["diagnostic_note"]
            == "High frequency with sufficient volume",
            high_frequency["diagnostic_note"]
            == "High frequency but limited credibility",
            high_frequency["diagnostic_note"]
            == "High loss ratio relative to portfolio",
        ],
        [
            "Review pricing relativities and underwriting rules",
            "Do not reprice directly; monitor or aggregate with similar segments",
            "Review premium adequacy, claim mix, and severity drivers",
        ],
        default="No immediate action; include in recurring monitoring",
    )

    return high_frequency.sort_values(
        ["frequency_index", "claims"],
        ascending=[False, False],
    ).head(top_n)


def add_rolling_frequency(
    table: pd.DataFrame,
    age_column: str,
    window: int = 5,
) -> pd.DataFrame:
    """
    Add a rolling exposure-weighted frequency for raw age diagnostics.
    """
    output = table.sort_values(age_column).copy()

    output[f"rolling_exposure_{window}"] = (
        output["exposure"].rolling(window=window, center=True, min_periods=1).sum()
    )

    output[f"rolling_claims_{window}"] = (
        output["claims"].rolling(window=window, center=True, min_periods=1).sum()
    )

    output[f"rolling_frequency_{window}"] = (
        output[f"rolling_claims_{window}"] / output[f"rolling_exposure_{window}"]
    )

    return output


def create_modeling_rationale() -> pd.DataFrame:
    """
    Document why the first benchmark model uses categorical treatment coding.
    """
    return pd.DataFrame(
        [
            {
                "variable": "territory",
                "raw_type": "nominal category",
                "eda_question": "Does claim frequency differ materially by geographic segment?",
                "benchmark_treatment": "categorical",
                "rationale": (
                    "Territory is nominal rather than ordinal. Treatment coding gives "
                    "interpretable relativities against a reference territory."
                ),
            },
            {
                "variable": "vehicle_type",
                "raw_type": "nominal category",
                "eda_question": "Does claim frequency differ by vehicle class?",
                "benchmark_treatment": "categorical",
                "rationale": (
                    "Vehicle type is nominal. Polynomial contrasts are not appropriate "
                    "because there is no natural ordering across Sedan, SUV, Truck, and Sports."
                ),
            },
            {
                "variable": "driver_age",
                "raw_type": "numeric",
                "eda_question": "Is the age-frequency relationship approximately log-linear?",
                "benchmark_treatment": "age bands",
                "rationale": (
                    "Driver age is available as a raw numeric variable, but frequency can "
                    "be nonlinear across ages. Age bands provide a transparent first benchmark; "
                    "continuous or spline specifications can be compared later."
                ),
            },
            {
                "variable": "vehicle_age",
                "raw_type": "numeric",
                "eda_question": "Is vehicle age approximately log-linear or better grouped?",
                "benchmark_treatment": "age bands",
                "rationale": (
                    "Vehicle age is numeric, but banding improves interpretability and "
                    "stability for a first diagnostic model. A later model can compare "
                    "continuous or spline-based specifications."
                ),
            },
        ]
    )


def round_numeric_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round numeric columns for cleaner exported tables.
    """
    output = df.copy()

    integer_columns = {
        "records",
        "policies",
        "claims",
        "distinct_policies",
    }

    two_decimal_columns = {
        "exposure",
        "total_claim_amount",
        "earned_premium",
        "average_claim_size",
        "pure_premium",
        "premium_per_exposure",
        "rolling_exposure_5",
        "rolling_claims_5",
    }

    six_decimal_columns = {
        "observed_frequency",
        "loss_ratio",
        "frequency_index",
        "loss_ratio_index",
        "rolling_frequency_5",
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


def save_line_chart(
    table: pd.DataFrame,
    x_column: str,
    y_column: str,
    y_label: str,
    file_name: str,
) -> None:
    """
    Save one line chart.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ax.plot(
        table[x_column],
        table[y_column],
        marker="o",
        linewidth=1.7,
        color=BLUE,
    )

    ax.set_xlabel(x_column.replace("_", " ").title())
    ax.set_ylabel(y_label)

    apply_axis_style(ax)

    fig.tight_layout()
    save_figure(fig, file_name)


def save_bar_chart(
    table: pd.DataFrame,
    x_column: str,
    y_column: str,
    y_label: str,
    file_name: str,
) -> None:
    """
    Save one bar chart.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    x_values = table[x_column].astype(str)

    ax.bar(
        x_values,
        table[y_column],
        color=BAR_BLUE,
        edgecolor="none",
    )

    ax.set_xlabel(x_column.replace("_", " ").title())
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=30)

    apply_axis_style(ax)
    ax.grid(True, axis="y", alpha=0.45)
    ax.grid(False, axis="x")

    fig.tight_layout()
    save_figure(fig, file_name)


def create_experience_outputs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Create experience review outputs.
    """
    portfolio_overview = create_portfolio_overview(df)

    segment_experience_by_territory_age = create_segment_experience(
        df,
        ["territory", "driver_age_band"],
    )

    segment_experience_by_vehicle_age = create_segment_experience(
        df,
        ["vehicle_type", "vehicle_age_band"],
    )

    segment_experience_by_territory_vehicle = create_segment_experience(
        df,
        ["territory", "vehicle_type"],
    )

    high_frequency_segments = create_high_frequency_segments(
        segment_experience_by_territory_age,
        top_n=15,
    )

    return {
        "portfolio_overview": round_numeric_outputs(portfolio_overview),
        "segment_experience_by_territory_age": round_numeric_outputs(
            segment_experience_by_territory_age
        ),
        "segment_experience_by_vehicle_age": round_numeric_outputs(
            segment_experience_by_vehicle_age
        ),
        "segment_experience_by_territory_vehicle": round_numeric_outputs(
            segment_experience_by_territory_vehicle
        ),
        "high_frequency_segments": round_numeric_outputs(high_frequency_segments),
    }


def create_exploratory_outputs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Create exploratory frequency diagnostics.
    """
    by_driver_age = add_rolling_frequency(
        create_frequency_table(df, ["driver_age"], min_exposure=20.0, min_claims=3),
        "driver_age",
        window=5,
    )

    by_driver_age_band = create_frequency_table(df, ["driver_age_band"])

    by_vehicle_age = add_rolling_frequency(
        create_frequency_table(df, ["vehicle_age"], min_exposure=20.0, min_claims=3),
        "vehicle_age",
        window=5,
    )

    by_vehicle_age_band = create_frequency_table(df, ["vehicle_age_band"])

    by_territory = create_frequency_table(df, ["territory"])

    by_vehicle_type = create_frequency_table(df, ["vehicle_type"])

    modeling_rationale = create_modeling_rationale()

    return {
        "eda_frequency_by_driver_age": round_numeric_outputs(by_driver_age),
        "eda_frequency_by_driver_age_band": round_numeric_outputs(by_driver_age_band),
        "eda_frequency_by_vehicle_age": round_numeric_outputs(by_vehicle_age),
        "eda_frequency_by_vehicle_age_band": round_numeric_outputs(by_vehicle_age_band),
        "eda_frequency_by_territory": round_numeric_outputs(by_territory),
        "eda_frequency_by_vehicle_type": round_numeric_outputs(by_vehicle_type),
        "eda_modeling_rationale": modeling_rationale,
    }


def write_experience_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Write experience review CSV outputs and Excel workbook.
    """
    write_csv_outputs(outputs)

    OUTPUT_EXCEL.mkdir(parents=True, exist_ok=True)
    excel_path = OUTPUT_EXCEL / "experience_review.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        outputs["portfolio_overview"].to_excel(
            writer,
            sheet_name="overview",
            index=False,
        )
        outputs["segment_experience_by_territory_age"].to_excel(
            writer,
            sheet_name="territory_age",
            index=False,
        )
        outputs["segment_experience_by_vehicle_age"].to_excel(
            writer,
            sheet_name="vehicle_age",
            index=False,
        )
        outputs["segment_experience_by_territory_vehicle"].to_excel(
            writer,
            sheet_name="territory_vehicle",
            index=False,
        )
        outputs["high_frequency_segments"].to_excel(
            writer,
            sheet_name="high_frequency",
            index=False,
        )


def write_exploratory_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Write exploratory diagnostics CSV outputs and Excel workbook.
    """
    write_csv_outputs(outputs)

    OUTPUT_EXCEL.mkdir(parents=True, exist_ok=True)
    excel_path = OUTPUT_EXCEL / "exploratory_frequency_review.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for name, table in outputs.items():
            sheet_name = name.replace("eda_", "")[:31]
            table.to_excel(writer, sheet_name=sheet_name, index=False)


def save_exploratory_charts(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save exploratory frequency charts.
    """
    save_line_chart(
        outputs["eda_frequency_by_driver_age"],
        x_column="driver_age",
        y_column="rolling_frequency_5",
        y_label="Rolling observed frequency",
        file_name="frequency_by_driver_age.png",
    )

    save_line_chart(
        outputs["eda_frequency_by_vehicle_age"],
        x_column="vehicle_age",
        y_column="rolling_frequency_5",
        y_label="Rolling observed frequency",
        file_name="frequency_by_vehicle_age.png",
    )

    save_bar_chart(
        outputs["eda_frequency_by_driver_age_band"],
        x_column="driver_age_band",
        y_column="frequency_index",
        y_label="Frequency index",
        file_name="frequency_index_by_driver_age_band.png",
    )

    save_bar_chart(
        outputs["eda_frequency_by_vehicle_age_band"],
        x_column="vehicle_age_band",
        y_column="frequency_index",
        y_label="Frequency index",
        file_name="frequency_index_by_vehicle_age_band.png",
    )

    save_bar_chart(
        outputs["eda_frequency_by_territory"],
        x_column="territory",
        y_column="frequency_index",
        y_label="Frequency index",
        file_name="frequency_index_by_territory.png",
    )


def run_experience_review(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Run the experience review layer.
    """
    outputs = create_experience_outputs(df)
    write_experience_outputs(outputs)

    print("Experience review outputs created:")
    for name in outputs:
        print(f"- outputs/tables/{name}.csv")
    print("- outputs/excel/experience_review.xlsx")

    return outputs


def run_exploratory_frequency(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Run the exploratory frequency diagnostics layer.
    """
    outputs = create_exploratory_outputs(df)
    write_exploratory_outputs(outputs)
    save_exploratory_charts(outputs)

    print("Exploratory frequency diagnostics created:")
    for name in outputs:
        print(f"- outputs/tables/{name}.csv")

    print("- outputs/excel/exploratory_frequency_review.xlsx")
    print("- outputs/figures/frequency_by_driver_age.png")
    print("- outputs/figures/frequency_by_vehicle_age.png")
    print("- outputs/figures/frequency_index_by_driver_age_band.png")
    print("- outputs/figures/frequency_index_by_vehicle_age_band.png")
    print("- outputs/figures/frequency_index_by_territory.png")

    return outputs


def main() -> None:
    """
    Run experience review and exploratory frequency diagnostics.
    """
    df = load_policy_data()

    run_experience_review(df)
    run_exploratory_frequency(df)


if __name__ == "__main__":
    main()