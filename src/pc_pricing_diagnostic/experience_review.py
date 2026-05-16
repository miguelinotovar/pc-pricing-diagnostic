from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "synthetic_policy_data.csv"
OUTPUT_TABLES = ROOT / "outputs" / "tables"
OUTPUT_EXCEL = ROOT / "outputs" / "excel"


def load_policy_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the synthetic policy-period dataset.

    The diagnostic works at policy-period level: each row represents one exposure
    record with exposure, claim count, premium, rating variables, and total claim
    amount for the period.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Policy data not found at {path}. "
            "Run synthetic_data.py before running the experience review."
        )

    df = pd.read_csv(path)

    required_columns = {
        "policy_id",
        "exposure",
        "claim_count",
        "total_claim_amount",
        "earned_premium",
        "territory",
        "driver_age_band",
        "vehicle_age_band",
        "vehicle_type",
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def create_portfolio_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a one-table portfolio overview.

    This is the highest-level diagnostic summary: exposure, claims, frequency,
    premium, loss ratio, pure premium, and average claim size.
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

    overview = pd.DataFrame(
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

    return overview


def create_segment_experience(
    df: pd.DataFrame,
    segment_columns: list[str],
    min_exposure: float = 100.0,
    min_claims: int = 20,
) -> pd.DataFrame:
    """
    Create segment-level experience.

    The table is designed to support pricing and underwriting review. It shows
    which segments have higher or lower claim frequency, loss ratio, pure premium,
    and whether the segment has enough volume to be interpreted.
    """
    total_frequency = df["claim_count"].sum() / df["exposure"].sum()
    total_loss_ratio = df["total_claim_amount"].sum() / df["earned_premium"].sum()

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


def create_high_frequency_segments(
    segment_experience: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Identify segments with high observed frequency relative to the portfolio average.

    This is an early commercial output: it points to segments that may deserve
    pricing, underwriting, or data quality review.
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


def round_numeric_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round numeric columns for cleaner exported tables.

    Counts are shown as integers. Monetary amounts and exposure-based amounts are
    shown with two decimals. Frequencies, ratios, and indexes keep six decimals
    because small differences may matter in diagnostic review.
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
    }

    six_decimal_columns = {
        "observed_frequency",
        "loss_ratio",
        "frequency_index",
        "loss_ratio_index",
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
    Write CSV outputs and one Excel workbook.

    CSV files are useful for reproducibility. The Excel workbook is useful as a
    client-facing review format.
    """
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_EXCEL.mkdir(parents=True, exist_ok=True)

    for name, table in outputs.items():
        table.to_csv(OUTPUT_TABLES / f"{name}.csv", index=False)

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


def main() -> None:
    df = load_policy_data()

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

    outputs = {
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

    write_outputs(outputs)

    print("Experience review outputs created:")
    for name in outputs:
        print(f"- outputs/tables/{name}.csv")

    print("- outputs/excel/experience_review.xlsx")


if __name__ == "__main__":
    main()