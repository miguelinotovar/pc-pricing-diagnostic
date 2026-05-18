from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

PIPELINE_MODULES = [
    "pc_pricing_diagnostic.synthetic_data",
    "pc_pricing_diagnostic.experience_review",
    "pc_pricing_diagnostic.exploratory_frequency",
    "pc_pricing_diagnostic.frequency_model",
    "pc_pricing_diagnostic.model_specification_comparison",
    "pc_pricing_diagnostic.frequency_model_diagnostics",
    "pc_pricing_diagnostic.executive_visuals",
]


def run_module(module_name: str) -> None:
    env = os.environ.copy()

    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{SRC}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(SRC)

    print(f"\nRunning {module_name}...", flush=True)

    subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    print("Starting P&C pricing diagnostic pipeline.", flush=True)

    for module_name in PIPELINE_MODULES:
        run_module(module_name)

    print("\nPipeline completed successfully.", flush=True)


if __name__ == "__main__":
    main()