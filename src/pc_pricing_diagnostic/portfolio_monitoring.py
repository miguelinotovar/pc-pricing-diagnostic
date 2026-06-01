import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pc_pricing_diagnostic.config import OUTPUT_EXCEL, OUTPUT_TABLES
from pc_pricing_diagnostic.io_utils import load_required_table, write_csv_outputs
from pc_pricing_diagnostic.plot_style import (
    GRID,
    NAVY,
    WARNING_COLORS,
    apply_axis_style,
    save_figure,
)


EXCEL_PATH = OUTPUT_EXCEL / "portfolio_monitoring_review.xlsx"

SEGMENT_KEYS = ["territory", "driver_age_band"]


def create_segment_label(df: pd.DataFrame) -> pd.Series:
    """
    Create readable segment labels for monitoring outputs.
    """
    return (
        df["territory"].astype(str)
        + " | "
        + df["driver_age_band"].astype(str)
    )


def build_reason(row: pd.Series) -> str:
    """
    Build a concise explanation for the warning level.
    """
    reasons = []

    if row["frequency_index"] >= 1.25:
        reasons.append("high frequency index")
    elif row["frequency_index"] >= 1.10:
        reasons.append("moderately elevated frequency index")

    if row["loss_ratio_index"] >= 1.25:
        reasons.append("high loss ratio index")
    elif row["loss_ratio_index"] >= 1.10:
        reasons.append("moderately elevated loss ratio index")

    if row["oe_ratio"] >= 1.20:
        reasons.append("observed claims materially above expected")
    elif row["oe_ratio"] >= 1.10:
        reasons.append("observed claims above expected")

    if row["credibility_flag"] != "Reviewable":
        reasons.append("limited credibility")

    if not reasons:
        return "within monitoring range"

    return "; ".join(reasons)


def recommended_action(warning_level: str) -> str:
    """
    Map warning levels to practical review actions.
    """
    actions = {
        "Escalate": (
            "Escalate to pricing and underwriting review; validate data before action"
        ),
        "Investigate": (
            "Investigate frequency, severity, premium adequacy, and segment mix"
        ),
        "Monitor": (
            "Monitor trend or aggregate with similar segments before acting"
        ),
        "No immediate action": (
            "No immediate action; include in recurring portfolio monitoring"
        ),
    }

    return actions[warning_level]


def create_monitoring_watchlist(
    segment_experience: pd.DataFrame,
    oe_by_segment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a portfolio monitoring watchlist by segment.

    The watchlist translates experience and model diagnostics into decision levels:
    Monitor, Investigate, Escalate, or No immediate action.
    """
    experience = segment_experience.copy()

    oe_columns = SEGMENT_KEYS + [
        "expected_claims",
        "expected_frequency",
        "oe_ratio",
        "claim_difference",
    ]

    oe = oe_by_segment[oe_columns].copy()

    watchlist = experience.merge(
        oe,
        on=SEGMENT_KEYS,
        how="left",
        validate="one_to_one",
    )

    watchlist["segment"] = create_segment_label(watchlist)
    watchlist["observed_claims"] = watchlist["claims"]
    watchlist["oe_ratio"] = watchlist["oe_ratio"].fillna(1.0)
    watchlist["expected_claims"] = watchlist["expected_claims"].fillna(
        watchlist["observed_claims"]
    )
    watchlist["claim_difference"] = watchlist["claim_difference"].fillna(0.0)

    reviewable = watchlist["credibility_flag"] == "Reviewable"

    escalate = reviewable & (
        (
            (watchlist["frequency_index"] >= 1.25)
            & (watchlist["loss_ratio_index"] >= 1.25)
        )
        | (
            (watchlist["loss_ratio_index"] >= 1.40)
            & (watchlist["oe_ratio"] >= 1.15)
        )
        | (
            (watchlist["frequency_index"] >= 1.35)
            & (watchlist["oe_ratio"] >= 1.20)
        )
    )

    investigate = reviewable & (
        (
            (watchlist["frequency_index"] >= 1.15)
            & (watchlist["loss_ratio_index"] >= 1.10)
        )
        | (watchlist["loss_ratio_index"] >= 1.25)
        | (watchlist["oe_ratio"] >= 1.15)
    )

    monitor = (
        (watchlist["frequency_index"] >= 1.10)
        | (watchlist["loss_ratio_index"] >= 1.10)
        | (watchlist["oe_ratio"] >= 1.10)
    )

    watchlist["warning_level"] = np.select(
        [escalate, investigate, monitor],
        ["Escalate", "Investigate", "Monitor"],
        default="No immediate action",
    )

    frequency_signal = (watchlist["frequency_index"] - 1.0).clip(lower=0.0)
    loss_ratio_signal = (watchlist["loss_ratio_index"] - 1.0).clip(lower=0.0)
    oe_signal = (watchlist["oe_ratio"] - 1.0).clip(lower=0.0)

    volume_factor = np.minimum(
        np.log1p(watchlist["exposure"]) / np.log1p(1000.0),
        1.0,
    )

    watchlist["priority_score"] = (
        0.35 * frequency_signal
        + 0.35 * loss_ratio_signal
        + 0.30 * oe_signal
    ) * (0.75 + 0.25 * volume_factor)

    watchlist["reason"] = watchlist.apply(build_reason, axis=1)
    watchlist["recommended_action"] = watchlist["warning_level"].map(
        recommended_action
    )

    output_columns = [
        "segment",
        "territory",
        "driver_age_band",
        "records",
        "policies",
        "exposure",
        "observed_claims",
        "expected_claims",
        "claim_difference",
        "observed_frequency",
        "expected_frequency",
        "frequency_index",
        "loss_ratio",
        "loss_ratio_index",
        "oe_ratio",
        "total_claim_amount",
        "earned_premium",
        "credibility_flag",
        "warning_level",
        "priority_score",
        "reason",
        "recommended_action",
    ]

    return (
        watchlist[output_columns]
        .sort_values(
            ["warning_level", "priority_score", "observed_claims"],
            ascending=[True, False, False],
        )
        .copy()
    )


def create_monitoring_summary(watchlist: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize portfolio monitoring results by warning level.
    """
    summary = (
        watchlist.groupby("warning_level", observed=True)
        .agg(
            segment_count=("segment", "count"),
            exposure=("exposure", "sum"),
            observed_claims=("observed_claims", "sum"),
            expected_claims=("expected_claims", "sum"),
            total_claim_amount=("total_claim_amount", "sum"),
            earned_premium=("earned_premium", "sum"),
            average_frequency_index=("frequency_index", "mean"),
            average_loss_ratio_index=("loss_ratio_index", "mean"),
            average_oe_ratio=("oe_ratio", "mean"),
            max_priority_score=("priority_score", "max"),
        )
        .reset_index()
    )

    summary["observed_frequency"] = (
        summary["observed_claims"] / summary["exposure"]
    )
    summary["expected_frequency"] = (
        summary["expected_claims"] / summary["exposure"]
    )
    summary["loss_ratio"] = (
        summary["total_claim_amount"] / summary["earned_premium"]
    )
    summary["total_oe_ratio"] = (
        summary["observed_claims"] / summary["expected_claims"]
    )

    warning_order = {
        "Escalate": 1,
        "Investigate": 2,
        "Monitor": 3,
        "No immediate action": 4,
    }

    summary["warning_order"] = summary["warning_level"].map(warning_order)

    return (
        summary.sort_values(
            ["warning_order", "max_priority_score", "observed_claims"],
            ascending=[True, False, False],
        )
        .drop(columns="warning_order")
        .copy()
    )


def create_monitoring_rationale() -> pd.DataFrame:
    """
    Explain monitoring levels and how to use them.
    """
    return pd.DataFrame(
        [
            {
                "warning_level": "Escalate",
                "interpretation": (
                    "Reviewable segment with multiple strong warning signals, such as "
                    "high frequency, high loss ratio pressure, or observed claims "
                    "materially above expected."
                ),
                "recommended_use": (
                    "Prioritize for pricing, underwriting, risk classification, or "
                    "data quality review."
                ),
            },
            {
                "warning_level": "Investigate",
                "interpretation": (
                    "Reviewable segment with at least one material warning signal."
                ),
                "recommended_use": (
                    "Investigate drivers before recommending rate action or rule changes."
                ),
            },
            {
                "warning_level": "Monitor",
                "interpretation": (
                    "Segment with moderate signal or limited credibility."
                ),
                "recommended_use": (
                    "Track over time or aggregate with similar segments before acting."
                ),
            },
            {
                "warning_level": "No immediate action",
                "interpretation": (
                    "Segment is within the current monitoring range."
                ),
                "recommended_use": (
                    "Include in recurring monitoring without immediate escalation."
                ),
            },
        ]
    )


def round_monitoring_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round monitoring outputs for cleaner exported tables.
    """
    output = df.copy()

    integer_columns = {
        "records",
        "policies",
        "observed_claims",
        "segment_count",
    }

    two_decimal_columns = {
        "exposure",
        "expected_claims",
        "claim_difference",
        "total_claim_amount",
        "earned_premium",
    }

    six_decimal_columns = {
        "observed_frequency",
        "expected_frequency",
        "frequency_index",
        "loss_ratio",
        "loss_ratio_index",
        "oe_ratio",
        "average_frequency_index",
        "average_loss_ratio_index",
        "average_oe_ratio",
        "max_priority_score",
        "total_oe_ratio",
        "priority_score",
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


def save_warning_matrix(watchlist: pd.DataFrame) -> None:
    """
    Save a frequency-index versus loss-ratio-index warning matrix.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for warning_level, group in watchlist.groupby("warning_level", observed=True):
        marker_size = 40 + 180 * (
            group["exposure"] / watchlist["exposure"].max()
        )

        ax.scatter(
            group["frequency_index"],
            group["loss_ratio_index"],
            s=marker_size,
            alpha=0.72,
            color=WARNING_COLORS[warning_level],
            label=warning_level,
            edgecolor="white",
            linewidth=0.6,
        )

    ax.axvline(1.0, color=NAVY, linewidth=1.0, alpha=0.75)
    ax.axhline(1.0, color=NAVY, linewidth=1.0, alpha=0.75)

    ax.set_xlabel("Frequency index (portfolio average = 1.00)")
    ax.set_ylabel("Loss ratio index (portfolio average = 1.00)")

    apply_axis_style(ax)

    legend = ax.legend(
        loc="upper left",
        fontsize=8,
        frameon=True,
        borderpad=0.4,
    )
    legend.get_frame().set_edgecolor(GRID)
    legend.get_frame().set_facecolor("white")

    fig.tight_layout()
    save_figure(fig, "portfolio_warning_matrix.png")


def save_top_watchlist_chart(watchlist: pd.DataFrame, top_n: int = 12) -> None:
    """
    Save a chart of top monitoring priorities.
    """
    chart_data = (
        watchlist.query("warning_level != 'No immediate action'")
        .sort_values("priority_score", ascending=False)
        .head(top_n)
        .iloc[::-1]
        .copy()
    )

    if chart_data.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    colors = chart_data["warning_level"].map(WARNING_COLORS)

    ax.barh(
        chart_data["segment"],
        chart_data["priority_score"],
        color=colors,
        edgecolor="none",
    )

    for index, value in enumerate(chart_data["priority_score"]):
        ax.text(
            value + 0.005,
            index,
            f"{value:.2f}",
            va="center",
            fontsize=7.5,
            color=NAVY,
        )

    ax.set_xlabel("Priority score")
    ax.set_ylabel("")
    ax.set_xlim(0, max(chart_data["priority_score"].max() * 1.20, 0.10))

    apply_axis_style(ax)
    ax.grid(True, axis="x", alpha=0.45, color=GRID, linewidth=0.7)
    ax.grid(False, axis="y")

    fig.tight_layout()
    save_figure(fig, "portfolio_monitoring_top_priorities.png")


def write_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Write monitoring outputs to CSV and Excel.
    """
    write_csv_outputs(outputs)
    OUTPUT_EXCEL.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        outputs["portfolio_monitoring_watchlist"].to_excel(
            writer,
            sheet_name="watchlist",
            index=False,
        )
        outputs["portfolio_monitoring_summary"].to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )
        outputs["portfolio_monitoring_rationale"].to_excel(
            writer,
            sheet_name="rationale",
            index=False,
        )


def main() -> None:
    segment_experience = load_required_table(
        "segment_experience_by_territory_age.csv"
    )
    oe_by_segment = load_required_table("frequency_oe_by_territory_age.csv")

    watchlist = create_monitoring_watchlist(
        segment_experience=segment_experience,
        oe_by_segment=oe_by_segment,
    )
    summary = create_monitoring_summary(watchlist)
    rationale = create_monitoring_rationale()

    outputs = {
        "portfolio_monitoring_watchlist": round_monitoring_outputs(watchlist),
        "portfolio_monitoring_summary": round_monitoring_outputs(summary),
        "portfolio_monitoring_rationale": rationale,
    }

    write_outputs(outputs)

    save_warning_matrix(watchlist)
    save_top_watchlist_chart(watchlist)

    print("Portfolio monitoring outputs created:")
    print("- outputs/tables/portfolio_monitoring_watchlist.csv")
    print("- outputs/tables/portfolio_monitoring_summary.csv")
    print("- outputs/tables/portfolio_monitoring_rationale.csv")
    print("- outputs/excel/portfolio_monitoring_review.xlsx")
    print("- outputs/figures/portfolio_warning_matrix.png")
    print("- outputs/figures/portfolio_monitoring_top_priorities.png")


if __name__ == "__main__":
    main()