from __future__ import annotations

import importlib


PIPELINE_MODULES = [
    "pc_pricing_diagnostic.synthetic_data",
    "pc_pricing_diagnostic.experience_diagnostics",
    "pc_pricing_diagnostic.frequency_analysis",
    "pc_pricing_diagnostic.portfolio_monitoring",
    "pc_pricing_diagnostic.executive_reporting",
]


def run_module(module_name: str) -> None:
    """
    Import and run one pipeline module.

    Each pipeline module is expected to expose a main() function.
    """
    print(f"\nRunning {module_name}...", flush=True)

    module = importlib.import_module(module_name)

    if not hasattr(module, "main"):
        raise AttributeError(f"Pipeline module does not expose main(): {module_name}")

    module.main()


def main() -> None:
    """
    Run the full P&C pricing diagnostic pipeline.
    """
    print("Starting P&C pricing diagnostic pipeline.", flush=True)

    for module_name in PIPELINE_MODULES:
        run_module(module_name)

    print("\nPipeline completed successfully.", flush=True)


if __name__ == "__main__":
    main()