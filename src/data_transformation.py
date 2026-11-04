import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.data_cleaning import clean_data


def transform_data(df):
    """Add time fields and business measures to cleaned data."""
    df = df.copy()

    required = [
        "order_date",
        "sales",
        "profit",
        "discount",
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Transformation columns missing: " + ", ".join(missing)
        )

    if df.empty:
        raise ValueError("Cannot transform an empty dataset.")

    if not pd.api.types.is_datetime64_any_dtype(df["order_date"]):
        raise TypeError(
            "order_date must be datetime. Run clean_data() first."
        )

    # 1. Derive time fields from the cleaned order date.
    # Recalculate existing source fields for consistency.
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["month_name"] = df["order_date"].dt.strftime("%b")
    df["quarter"] = df["order_date"].dt.quarter

    # YYYY-MM sorts in chronological order across different years.
    df["year_month"] = (
        df["order_date"].dt.to_period("M").astype("string")
    )

    # 2. Sales already represents the recorded discounted amount.
    # Do not multiply by quantity or apply the discount again.
    df["revenue"] = df["sales"]

    # 3. Calculate each row's profit margin.
    # For zero revenue, margin is undefined, so keep it missing.
    df["profit_margin_pct"] = (
        df["profit"]
        .div(df["revenue"].replace(0, np.nan))
        .mul(100)
    )

    # 4. Group discount rates into readable ranges.
    # With right=True:
    # 10% belongs to Low, and 20% belongs to Medium.
    df["discount_category"] = pd.cut(
        df["discount"],
        bins=[-0.001, 0, 0.10, 0.20, 1.0],
        labels=[
            "No Discount",
            "Low (0-10%]",
            "Medium (10-20%]",
            "High (>20%)",
        ],
        right=True,
        include_lowest=True,
        ordered=True,
    )

    # Retain the cleaning audit for the final report.
    df.attrs["cleaning_summary"] = dict(
        df.attrs.get("cleaning_summary", {})
    )

    print("\n--- TRANSFORMATION SUMMARY ---")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print(
        "Date range:",
        df["order_date"].min().date(),
        "to",
        df["order_date"].max().date(),
    )

    print("\nDiscount category counts:")
    print(
        df["discount_category"]
        .value_counts(sort=False)
        .to_string()
    )

    print(
        "\nRows with undefined profit margin:",
        int(df["profit_margin_pct"].isna().sum()),
    )

    print("\nPreview of derived columns:")
    preview_columns = [
        "order_id",
        "year_month",
        "revenue",
        "profit",
        "profit_margin_pct",
        "discount_category",
    ]

    print(
        df[preview_columns]
        .head()
        .to_string(index=False)
    )

    return df


if __name__ == "__main__":
    raw_data = load_data()
    cleaned_data = clean_data(raw_data)
    transformed_data = transform_data(cleaned_data)

    # Check that transformation preserved records and amounts.
    if len(transformed_data) != len(cleaned_data):
        raise ValueError("Transformation unexpectedly changed row count.")

    if not transformed_data["revenue"].equals(cleaned_data["sales"]):
        raise ValueError("Revenue does not match the cleaned sales amounts.")

    if transformed_data["discount_category"].isna().any():
        raise ValueError("Some discounts were not assigned a category.")

    print("\nData transformation check passed.")
    print("Row count and recorded sales amounts are preserved.")