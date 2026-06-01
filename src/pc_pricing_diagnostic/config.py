from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DATA = DATA_DIR / "raw"
PROCESSED_DATA = DATA_DIR / "processed"

OUTPUTS = ROOT / "outputs"
OUTPUT_TABLES = OUTPUTS / "tables"
OUTPUT_FIGURES = OUTPUTS / "figures"
OUTPUT_EXCEL = OUTPUTS / "excel"

COMMERCIAL = ROOT / "commercial"