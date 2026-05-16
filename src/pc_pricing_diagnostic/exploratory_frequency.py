from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "synthetic_policy_data.csv"
OUTPUT_TABLES = ROOT / "outputs" / "tables"
OUTPUT_FIGURES = ROOT / "outputs" / "figures"
OUTPUT_EXCEL = ROOT / "outputs" / "excel"


DRIVER_AGE_BAND_ORDER = ["16-24", "25-39", "40-64", "65+"]
VEHICLE_AGE_BAND_ORDER = ["0-3", "4-7", "8-12", "13+"]
TERRITORY_ORDER = ["Rural", "Suburban", "Urban B", "Urban A"]
VEHICLE_TYPE_ORDER = ["Sedan", "SUV", "Truck", "Sports"]


def load_policy_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the synthetic policy-period dataset.

    This EDA module assumes the current standardized schema produced by
    synthetic_data.py. Each row is a policy exposure record.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Policy data not found at {path}. "
            "Run synthetic_data.py before running exploratory diagnostics."
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
    Apply explicit category order for cleaner EDA tables and charts.
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
    Compute portfolio-level observed frequency per unit of exposure.
    """
    return df["claim_count"].sum() / df["exposure"].sum()


def create_frequency_table(
    df: pd.DataFrame,
    group_columns: list[str],
    min_exposure: float = 100.0,
    min_claims: int = 20,
) -> pd.DataFrame:
    """
    Create EDA frequency and loss-ratio diagnostics by group.

    This table is used to decide whether a variable should enter the first
    benchmark model as categorical, continuous, or as a candidate for later
    spline/nonlinear treatment.
    """
    total_frequency = portfolio_frequency(df)
    total_loss_ratio = df["total_claim_amount"].sum() / df["earned_premium"].sum()

    summary = (
        df.groupby(group_columns, observed=True)
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

    return summary


def add_rolling_frequency(
    table: pd.DataFrame,
    age_column: str,
    window: int = 5,
) -> pd.DataFrame:
    """
    Add a rolling exposure-weighted frequency for raw age EDA.

    Raw age-level frequency can be noisy. A rolling view helps assess whether the
    relationship is roughly linear, nonlinear, or better handled through groups.
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

    This is not a substitute for full model selection. It records the modeling logic
    behind the initial benchmark specification.
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
    title: str,
    y_label: str,
    file_name: str,
) -> None:
    """
    Save one line chart.
    """
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(table[x_column], table[y_column], marker="o", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(x_column.replace("_", " ").title())
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(OUTPUT_FIGURES / file_name, dpi=150)
    plt.close(fig)


def save_bar_chart(
    table: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    y_label: str,
    file_name: str,
) -> None:
    """
    Save one bar chart.
    """
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    x_values = table[x_column].astype(str)
    ax.bar(x_values, table[y_column])
    ax.set_title(title)
    ax.set_xlabel(x_column.replace("_", " ").title())
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    fig.savefig(OUTPUT_FIGURES / file_name, dpi=150)
    plt.close(fig)


def write_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Write EDA outputs to CSV and Excel.
    """
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_EXCEL.mkdir(parents=True, exist_ok=True)

    for name, table in outputs.items():
        table.to_csv(OUTPUT_TABLES / f"{name}.csv", index=False)

    excel_path = OUTPUT_EXCEL / "exploratory_frequency_review.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for name, table in outputs.items():
            sheet_name = name.replace("eda_", "")[:31]
            table.to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> None:
    df = load_policy_data()

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

    outputs = {
        "eda_frequency_by_driver_age": round_numeric_outputs(by_driver_age),
        "eda_frequency_by_driver_age_band": round_numeric_outputs(by_driver_age_band),
        "eda_frequency_by_vehicle_age": round_numeric_outputs(by_vehicle_age),
        "eda_frequency_by_vehicle_age_band": round_numeric_outputs(by_vehicle_age_band),
        "eda_frequency_by_territory": round_numeric_outputs(by_territory),
        "eda_frequency_by_vehicle_type": round_numeric_outputs(by_vehicle_type),
        "eda_modeling_rationale": modeling_rationale,
    }

    write_outputs(outputs)

    save_line_chart(
        by_driver_age,
        x_column="driver_age",
        y_column="rolling_frequency_5",
        title="Rolling Claim Frequency by Driver Age",
        y_label="Rolling observed frequency",
        file_name="frequency_by_driver_age.png",
    )

    save_line_chart(
        by_vehicle_age,
        x_column="vehicle_age",
        y_column="rolling_frequency_5",
        title="Rolling Claim Frequency by Vehicle Age",
        y_label="Rolling observed frequency",
        file_name="frequency_by_vehicle_age.png",
    )

    save_bar_chart(
        by_driver_age_band,
        x_column="driver_age_band",
        y_column="frequency_index",
        title="Frequency Index by Driver Age Band",
        y_label="Frequency index",
        file_name="frequency_index_by_driver_age_band.png",
    )

    save_bar_chart(
        by_vehicle_age_band,
        x_column="vehicle_age_band",
        y_column="frequency_index",
        title="Frequency Index by Vehicle Age Band",
        y_label="Frequency index",
        file_name="frequency_index_by_vehicle_age_band.png",
    )

    save_bar_chart(
        by_territory,
        x_column="territory",
        y_column="frequency_index",
        title="Frequency Index by Territory",
        y_label="Frequency index",
        file_name="frequency_index_by_territory.png",
    )

    print("Exploratory frequency diagnostics created:")
    for name in outputs:
        print(f"- outputs/tables/{name}.csv")

    print("- outputs/excel/exploratory_frequency_review.xlsx")
    print("- outputs/figures/frequency_by_driver_age.png")
    print("- outputs/figures/frequency_by_vehicle_age.png")
    print("- outputs/figures/frequency_index_by_driver_age_band.png")
    print("- outputs/figures/frequency_index_by_vehicle_age_band.png")
    print("- outputs/figures/frequency_index_by_territory.png")


if __name__ == "__main__":
    main()