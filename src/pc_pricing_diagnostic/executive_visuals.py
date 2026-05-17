import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pc_pricing_diagnostic.frequency_model import (
    OUTPUT_EXCEL,
    OUTPUT_TABLES,
    ROOT,
)


OUTPUT_FIGURES = ROOT / "outputs" / "figures"
EXECUTIVE_EXCEL_PATH = OUTPUT_EXCEL / "executive_visual_pack.xlsx"


def load_required_table(file_name: str) -> pd.DataFrame:
    """
    Load a required CSV table from outputs/tables.

    Raise a clear error if the file does not exist.
    """
    path = OUTPUT_TABLES / file_name

    if not path.exists():
        raise FileNotFoundError(f"Required table not found: {path}")

    return pd.read_csv(path)


def prepare_segment_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a readable segment label for charts.

    The current executive visuals focus on territory-age combinations because
    those are easy to explain in pricing conversations.
    """
    output = df.copy()

    output["segment_label"] = (
        output["territory"].astype(str)
        + " | "
        + output["driver_age_band"].astype(str)
    )

    return output


def prettify_model_name(model_name: str) -> str:
    """
    Convert internal model names to cleaner chart labels.
    """
    mapping = {
        "null_offset_only": "Null offset only",
        "categorical_age_bands": "Categorical age bands",
        "continuous_linear_age": "Continuous linear age",
        "continuous_quadratic_age": "Continuous quadratic age",
        "hybrid_driver_band_vehicle_quadratic": "Hybrid age specification",
    }

    return mapping.get(model_name, model_name)


def save_top_frequency_segments_chart(high_frequency_segments: pd.DataFrame) -> None:
    """
    Save a horizontal bar chart of the top segments by frequency index.

    This is one of the most commercial charts because it highlights segments
    that may deserve pricing or underwriting review.
    """
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

    chart_data = prepare_segment_label(high_frequency_segments)

    chart_data = chart_data.sort_values(
        ["frequency_index", "claims"],
        ascending=[False, False],
    ).head(10)

    chart_data = chart_data.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        chart_data["segment_label"],
        chart_data["frequency_index"],
    )
    ax.axvline(1.0, linewidth=1)

    ax.set_title("Top Segments by Frequency Index")
    ax.set_xlabel("Frequency index vs portfolio average")
    ax.set_ylabel("Segment")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    fig.savefig(
        OUTPUT_FIGURES / "executive_top_frequency_segments.png",
        dpi=150,
    )
    plt.close(fig)


def save_top_loss_ratio_segments_chart(high_frequency_segments: pd.DataFrame) -> None:
    """
    Save a horizontal bar chart of the top segments by loss ratio index.

    This helps explain which segments may be commercially important from a
    premium adequacy perspective, not only a claim frequency perspective.
    """
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

    chart_data = prepare_segment_label(high_frequency_segments)

    chart_data = chart_data.sort_values(
        ["loss_ratio_index", "claims"],
        ascending=[False, False],
    ).head(10)

    chart_data = chart_data.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        chart_data["segment_label"],
        chart_data["loss_ratio_index"],
    )
    ax.axvline(1.0, linewidth=1)

    ax.set_title("Top Segments by Loss Ratio Index")
    ax.set_xlabel("Loss ratio index vs portfolio average")
    ax.set_ylabel("Segment")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    fig.savefig(
        OUTPUT_FIGURES / "executive_top_loss_ratio_segments.png",
        dpi=150,
    )
    plt.close(fig)


def save_calibration_chart(calibration_by_decile: pd.DataFrame) -> None:
    """
    Save observed vs expected frequency by risk decile.

    This is the key model validation chart that is still understandable in a
    commercial conversation.
    """
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

    chart_data = calibration_by_decile.sort_values("risk_decile").copy()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        chart_data["risk_decile"],
        chart_data["observed_frequency"],
        marker="o",
        label="Observed frequency",
    )
    ax.plot(
        chart_data["risk_decile"],
        chart_data["expected_frequency"],
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
        OUTPUT_FIGURES / "executive_calibration_by_decile.png",
        dpi=150,
    )
    plt.close(fig)


def save_model_specification_chart(
    model_specification_ranking: pd.DataFrame,
) -> None:
    """
    Save a chart comparing candidate model specifications by test mean Poisson deviance.

    This is slightly more technical, but useful as a credibility chart when a
    pricing manager or technical reviewer wants to see why one benchmark model
    was preferred.
    """
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

    chart_data = model_specification_ranking.sort_values("rank").copy()
    chart_data["model_label"] = chart_data["model"].map(prettify_model_name)

    chart_data = chart_data.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        chart_data["model_label"],
        chart_data["mean_poisson_deviance"],
    )

    ax.set_title("Model Specification Comparison")
    ax.set_xlabel("Test mean Poisson deviance")
    ax.set_ylabel("Model specification")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    fig.savefig(
        OUTPUT_FIGURES / "executive_model_specification_comparison.png",
        dpi=150,
    )
    plt.close(fig)


def create_visual_inventory() -> pd.DataFrame:
    """
    Document the purpose of each executive visual.
    """
    return pd.DataFrame(
        [
            {
                "file_name": "executive_top_frequency_segments.png",
                "purpose": "Show segments with elevated claim frequency relative to the portfolio.",
                "primary_audience": "Commercial and technical",
                "use_case": "Supports pricing and underwriting prioritization discussions.",
            },
            {
                "file_name": "executive_top_loss_ratio_segments.png",
                "purpose": "Show segments with elevated loss ratio relative to the portfolio.",
                "primary_audience": "Commercial and technical",
                "use_case": "Supports premium adequacy and profitability discussions.",
            },
            {
                "file_name": "executive_calibration_by_decile.png",
                "purpose": "Show whether observed and expected frequencies align across risk deciles.",
                "primary_audience": "Technical and mixed audience",
                "use_case": "Supports model validation and trust in the benchmark model.",
            },
            {
                "file_name": "executive_model_specification_comparison.png",
                "purpose": "Compare alternative model specifications using held-out test performance.",
                "primary_audience": "Technical reviewer",
                "use_case": "Supports explanation of why the benchmark specification was selected.",
            },
        ]
    )


def write_excel_pack(
    high_frequency_segments: pd.DataFrame,
    calibration_by_decile: pd.DataFrame,
    model_specification_ranking: pd.DataFrame,
    visual_inventory: pd.DataFrame,
) -> None:
    """
    Write a compact Excel pack with the source tables behind the executive visuals.
    """
    OUTPUT_EXCEL.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(EXECUTIVE_EXCEL_PATH, engine="openpyxl") as writer:
        high_frequency_segments.to_excel(
            writer,
            sheet_name="high_frequency_segments",
            index=False,
        )
        calibration_by_decile.to_excel(
            writer,
            sheet_name="calibration_by_decile",
            index=False,
        )
        model_specification_ranking.to_excel(
            writer,
            sheet_name="model_spec_ranking",
            index=False,
        )
        visual_inventory.to_excel(
            writer,
            sheet_name="visual_inventory",
            index=False,
        )


def main() -> None:
    high_frequency_segments = load_required_table("high_frequency_segments.csv")
    calibration_by_decile = load_required_table(
        "frequency_model_calibration_by_decile.csv"
    )
    model_specification_ranking = load_required_table(
        "frequency_model_specification_ranking.csv"
    )

    visual_inventory = create_visual_inventory()

    save_top_frequency_segments_chart(high_frequency_segments)
    save_top_loss_ratio_segments_chart(high_frequency_segments)
    save_calibration_chart(calibration_by_decile)
    save_model_specification_chart(model_specification_ranking)

    write_excel_pack(
        high_frequency_segments=high_frequency_segments,
        calibration_by_decile=calibration_by_decile,
        model_specification_ranking=model_specification_ranking,
        visual_inventory=visual_inventory,
    )

    visual_inventory.to_csv(
        OUTPUT_TABLES / "executive_visual_inventory.csv",
        index=False,
    )

    print("Executive visuals created:")
    print("- outputs/figures/executive_top_frequency_segments.png")
    print("- outputs/figures/executive_top_loss_ratio_segments.png")
    print("- outputs/figures/executive_calibration_by_decile.png")
    print("- outputs/figures/executive_model_specification_comparison.png")
    print("- outputs/tables/executive_visual_inventory.csv")
    print("- outputs/excel/executive_visual_pack.xlsx")


if __name__ == "__main__":
    main()