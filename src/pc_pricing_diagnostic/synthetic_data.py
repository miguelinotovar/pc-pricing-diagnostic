from pathlib import Path

import numpy as np
import pandas as pd


# Repository root.
# __file__ is provided automatically by Python when this script is executed.
# For this file:
#   src/pc_pricing_diagnostic/synthetic_data.py
# parents[2] points to:
#   pc-pricing-diagnostic/
ROOT = Path(__file__).resolve().parents[2]

DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUT_TABLES = ROOT / "outputs" / "tables"


def calculate_age(birth_dates: pd.Series, valuation_date: pd.Timestamp) -> pd.Series:
    """
    Calculate age at a valuation date from dates of birth.

    This mirrors a common insurance workflow:
    raw date of birth -> attained age -> pricing/diagnostic age band.
    """
    birth_dates = pd.to_datetime(birth_dates)

    age = valuation_date.year - birth_dates.dt.year

    had_birthday = (
        (birth_dates.dt.month < valuation_date.month)
        | (
            (birth_dates.dt.month == valuation_date.month)
            & (birth_dates.dt.day <= valuation_date.day)
        )
    )

    return age - (~had_birthday).astype(int)


def sample_driver_ages(n_policies: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample discrete driver ages directly.

    The distribution is intentionally shaped to resemble a simplified motor portfolio:
    most drivers are middle-aged, with fewer very young and elderly drivers.

    Age bands are derived later from the raw age variable.
    """
    possible_ages = np.arange(16, 86)

    weights = np.ones_like(possible_ages, dtype=float)
    weights[(possible_ages >= 16) & (possible_ages <= 24)] = 0.65
    weights[(possible_ages >= 25) & (possible_ages <= 39)] = 1.35
    weights[(possible_ages >= 40) & (possible_ages <= 64)] = 1.55
    weights[possible_ages >= 65] = 0.55

    weights = weights / weights.sum()

    return rng.choice(possible_ages, size=n_policies, p=weights)


def generate_birth_dates(
    target_ages: np.ndarray,
    valuation_date: pd.Timestamp,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate dates of birth consistent with sampled driver ages.

    The generated date of birth is constructed so that the derived age at the
    valuation date matches the sampled target age.
    """
    days_since_last_birthday = rng.integers(0, 365, size=len(target_ages))

    birth_dates = []

    for age, days in zip(target_ages, days_since_last_birthday):
        last_birthday = valuation_date - pd.DateOffset(years=int(age))
        birth_date = last_birthday - pd.Timedelta(days=int(days))
        birth_dates.append(birth_date.date().isoformat())

    return np.array(birth_dates)


def sample_vehicle_model_years(
    n_policies: int,
    valuation_date: pd.Timestamp,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample vehicle model years directly.

    Vehicle age is derived later as:
        valuation year - model year

    This is more realistic than generating vehicle age bands directly, because
    model year is commonly available in policy data, while age bands are derived
    for pricing, underwriting, or diagnostic reporting.
    """
    valuation_year = valuation_date.year

    possible_vehicle_ages = np.arange(0, 21)

    weights = np.ones_like(possible_vehicle_ages, dtype=float)
    weights[(possible_vehicle_ages >= 0) & (possible_vehicle_ages <= 3)] = 1.15
    weights[(possible_vehicle_ages >= 4) & (possible_vehicle_ages <= 7)] = 1.35
    weights[(possible_vehicle_ages >= 8) & (possible_vehicle_ages <= 12)] = 1.10
    weights[possible_vehicle_ages >= 13] = 0.75

    weights = weights / weights.sum()

    vehicle_age = rng.choice(possible_vehicle_ages, size=n_policies, p=weights)
    vehicle_model_year = valuation_year - vehicle_age

    return vehicle_model_year


def generate_synthetic_portfolio(
    n_policies: int = 25_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic P&C motor-style portfolio.

    The data is not intended to represent any real insurer. It is designed to
    demonstrate a pricing diagnostic workflow:

    - exposure-adjusted claim frequency;
    - segment-level experience review;
    - observed vs. expected analysis;
    - benchmark Poisson modeling;
    - identification of segments requiring pricing or underwriting review.
    """
    rng = np.random.default_rng(seed)
    valuation_date = pd.Timestamp("2026-01-01")

    # Policy-level exposure.
    # A value of 1.0 means one full policy-year.
    # Partial exposures force the diagnostic to adjust claim counts by exposure.
    exposure = rng.uniform(0.10, 1.00, size=n_policies).round(4)

    # Core rating / underwriting variables.
    territory = rng.choice(
        ["Urban A", "Urban B", "Suburban", "Rural"],
        size=n_policies,
        p=[0.25, 0.25, 0.30, 0.20],
    )

    vehicle_type = rng.choice(
        ["Sedan", "SUV", "Truck", "Sports"],
        size=n_policies,
        p=[0.48, 0.32, 0.15, 0.05],
    )

    # Raw driver attributes.
    # Date of birth is generated first; age and age bands are derived.
    target_driver_age = sample_driver_ages(n_policies, rng)

    driver_birth_date = generate_birth_dates(
        target_ages=target_driver_age,
        valuation_date=valuation_date,
        rng=rng,
    )

    driver_age = calculate_age(
        birth_dates=pd.Series(driver_birth_date),
        valuation_date=valuation_date,
    ).to_numpy()

    driver_age_band = pd.cut(
        driver_age,
        bins=[15, 24, 39, 64, 120],
        labels=["16-24", "25-39", "40-64", "65+"],
        right=True,
    ).astype(str)

    # Raw vehicle attributes.
    # Model year is generated first; age and age bands are derived.
    vehicle_model_year = sample_vehicle_model_years(
        n_policies=n_policies,
        valuation_date=valuation_date,
        rng=rng,
    )

    vehicle_age = valuation_date.year - vehicle_model_year

    vehicle_age_band = pd.cut(
        vehicle_age,
        bins=[-1, 3, 7, 12, 100],
        labels=["0-3", "4-7", "8-12", "13+"],
        right=True,
    ).astype(str)

    # Baseline annual claim frequency per unit of exposure.
    # A value of 0.065 means 6.5 expected claims per 100 policy-years
    # before applying risk relativities.
    base_frequency = 0.065

    # Multiplicative risk relativities used to generate the hidden true frequency.
    # Values above 1 increase expected claim frequency; values below 1 reduce it.
    # These relativities are intentionally simplified and known only because this
    # is synthetic data.
    territory_factor = {
        "Urban A": 1.45,
        "Urban B": 1.15,
        "Suburban": 0.90,
        "Rural": 0.70,
    }

    driver_factor = {
        "16-24": 1.90,
        "25-39": 1.15,
        "40-64": 0.85,
        "65+": 1.25,
    }

    vehicle_age_factor = {
        "0-3": 0.85,
        "4-7": 1.00,
        "8-12": 1.15,
        "13+": 1.35,
    }

    vehicle_type_factor = {
        "Sedan": 0.90,
        "SUV": 1.00,
        "Truck": 1.20,
        "Sports": 1.85,
    }

    # Start every policy at the baseline frequency.
    # Then multiply by each applicable risk factor to obtain the policy-level
    # true annual claim frequency.
    true_frequency = np.full(n_policies, base_frequency)

    for key, value in territory_factor.items():
        true_frequency *= np.where(territory == key, value, 1.0)

    for key, value in driver_factor.items():
        true_frequency *= np.where(driver_age_band == key, value, 1.0)

    for key, value in vehicle_age_factor.items():
        true_frequency *= np.where(vehicle_age_band == key, value, 1.0)

    for key, value in vehicle_type_factor.items():
        true_frequency *= np.where(vehicle_type == key, value, 1.0)

    # Add interaction effects that are intentionally not reflected in the simple
    # premium proxy below. This creates realistic diagnostic tension:
    # some segments will appear misaligned even if the main effects are reasonable.
    true_frequency *= np.where(
        (territory == "Urban A") & (driver_age_band == "16-24"),
        1.25,
        1.0,
    )

    true_frequency *= np.where(
        (vehicle_type == "Sports") & (driver_age_band == "16-24"),
        1.20,
        1.0,
    )

    # Expected claims are exposure times annual frequency.
    # Claim counts are simulated from a Poisson process.
    expected_claims_true = exposure * true_frequency
    claim_count = rng.poisson(expected_claims_true)

    # Claim amount simulation.
    # The first MVP focuses on claim frequency, but claim amount is included so
    # later diagnostics can also discuss severity and loss ratio.
    severity_factor = np.ones(n_policies)
    severity_factor *= np.where(vehicle_type == "Sports", 1.35, 1.0)
    severity_factor *= np.where(vehicle_type == "Truck", 1.15, 1.0)
    severity_factor *= np.where(territory == "Urban A", 1.10, 1.0)

    avg_claim_size = 2_800 * severity_factor

    total_claim_amount = np.zeros(n_policies)
    positive_claims = claim_count > 0

    total_claim_amount[positive_claims] = rng.gamma(
        shape=2.0 * claim_count[positive_claims],
        scale=avg_claim_size[positive_claims] / 2.0,
    )

    # Simple earned premium proxy.
    # It intentionally captures only part of the true risk structure.
    # This lets the sample diagnostic identify possible pricing gaps.
    pricing_factor = np.ones(n_policies)
    pricing_factor *= np.where(territory == "Urban A", 1.25, 1.0)
    pricing_factor *= np.where(territory == "Urban B", 1.10, 1.0)
    pricing_factor *= np.where(driver_age_band == "16-24", 1.45, 1.0)
    pricing_factor *= np.where(driver_age_band == "65+", 1.10, 1.0)
    pricing_factor *= np.where(vehicle_type == "Sports", 1.45, 1.0)
    pricing_factor *= np.where(vehicle_type == "Truck", 1.15, 1.0)

    earned_premium = exposure * 750 * pricing_factor

    return pd.DataFrame(
        {
            "policy_id": np.arange(1, n_policies + 1),
            "valuation_date": valuation_date.date().isoformat(),
            "driver_birth_date": driver_birth_date,
            "driver_age": driver_age,
            "driver_age_band": driver_age_band,
            "vehicle_model_year": vehicle_model_year,
            "vehicle_age": vehicle_age,
            "vehicle_age_band": vehicle_age_band,
            "territory": territory,
            "vehicle_type": vehicle_type,
            "exposure": exposure,
            "claim_count": claim_count,
            "total_claim_amount": total_claim_amount.round(2),
            "earned_premium": earned_premium.round(2),
            # Hidden synthetic truth.
            # These fields are useful for testing the pipeline, but they would not
            # exist in a real client dataset.
            "true_annual_frequency": true_frequency.round(6),
            "expected_claims_true": expected_claims_true.round(6),
        }
    )


def create_portfolio_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a first experience review by territory and driver age band.

    This is not the final diagnostic. It is the first client-facing style output:
    exposure, claims, observed frequency, claim amount, earned premium, and loss
    ratio by segment.
    """
    summary = (
        df.groupby(["territory", "driver_age_band"], observed=True)
        .agg(
            policies=("policy_id", "count"),
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
        0.0,
    )

    summary["loss_ratio"] = summary["total_claim_amount"] / summary["earned_premium"]

    return summary.sort_values(
        ["observed_frequency", "claims"],
        ascending=[False, False],
    )


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_portfolio()
    summary = create_portfolio_summary(df)

    data_path = DATA_PROCESSED / "synthetic_policy_data.csv"
    summary_path = OUTPUT_TABLES / "synthetic_portfolio_summary.csv"

    df.to_csv(data_path, index=False)
    summary.to_csv(summary_path, index=False)

    total_exposure = df["exposure"].sum()
    total_claims = df["claim_count"].sum()
    observed_frequency = total_claims / total_exposure
    total_total_claim_amount = df["total_claim_amount"].sum()
    total_earned_premium = df["earned_premium"].sum()
    loss_ratio = total_total_claim_amount / total_earned_premium

    print(f"Saved policy data to: {data_path}")
    print(f"Saved summary table to: {summary_path}")
    print(f"Rows: {len(df):,}")
    print(f"Total exposure: {total_exposure:,.2f}")
    print(f"Total claims: {total_claims:,}")
    print(f"Observed frequency: {observed_frequency:.4f}")
    print(f"Total claim amount: {total_total_claim_amount:,.2f}")
    print(f"Total earned premium: {total_earned_premium:,.2f}")
    print(f"Loss ratio: {loss_ratio:.4f}")


if __name__ == "__main__":
    main()