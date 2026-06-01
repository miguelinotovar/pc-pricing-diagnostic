from pathlib import Path

import pandas as pd

from pc_pricing_diagnostic.config import OUTPUT_TABLES


def load_required_table(
    file_name: str,
    base_dir: Path = OUTPUT_TABLES,
) -> pd.DataFrame:
    """
    Load a required CSV table.

    Raise a clear error if the file does not exist.
    """
    path = base_dir / file_name

    if not path.exists():
        raise FileNotFoundError(f"Required table not found: {path}")

    return pd.read_csv(path)


def write_csv_outputs(
    outputs: dict[str, pd.DataFrame],
    output_dir: Path = OUTPUT_TABLES,
) -> None:
    """
    Write a dictionary of DataFrames to CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)