import numpy as np
import pandas as pd

from src.config import (
    REQUIRED_COLUMNS,
    DATE_FORMAT,
    DISCOUNT_FORMAT,
)
from src.data_loader import load_data


def clean_data(df):
    """Clean transaction data and retain an audit summary."""
    df = df.copy()
    summary = {"input_rows": len(df)}

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns missing: " + ", ".join(missing_columns)
        )

    # 1. Remove exact duplicate rows.
    summary["exact_duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # 2. Trim text and standardize empty values.
    text_columns = [
        "order_id",
        "customer_id",
        "product",
        "category",
        "region",
        "payment_method",
        "order_status",
    ]

    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()

        missing_text = (
            df[column]
            .str.lower()
            .isin(["", "nan", "none", "null", "n/a"])
        )

        df.loc[missing_text, column] = pd.NA

    # 3. Parse dates using the known source format.
    df["order_date"] = pd.to_datetime(
        df["order_date"],
        format=DATE_FORMAT,
        errors="coerce",
    )

    # 4. Convert numerical columns.
    numeric_columns = ["sales", "quantity", "discount", "profit"]

    # Preserve useful optional source measures as numbers too.
    for column in [
        "unit_price",
        "cost",
        "shipping_cost",
        "shipping_days",
        "source_profit_margin_pct",
    ]:
        if column in df.columns:
            numeric_columns.append(column)

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    # 5. Remove rows missing essential transaction information.
    essential_columns = [
        "order_id",
        "order_date",
        "product",
        "sales",
        "quantity",
    ]

    invalid_essential = df[essential_columns].isna().any(axis=1)

    summary["missing_or_invalid_essential_rows_removed"] = int(
        invalid_essential.sum()
    )

    df = df.loc[~invalid_essential].copy()

    # 6. Validate transaction values.
    # Negative profit is valid and must be retained.
    invalid_transaction = (
        (df["sales"] < 0)
        | (df["quantity"] <= 0)
        | (df["quantity"] % 1 != 0)
    )

    summary["invalid_transaction_rows_removed"] = int(
        invalid_transaction.sum()
    )

    df = df.loc[~invalid_transaction].copy()
    df["quantity"] = df["quantity"].astype("int64")

    # 7. Fill missing descriptive fields.
    categorical_columns = [
        "customer_id",
        "category",
        "region",
        "payment_method",
        "order_status",
    ]

    summary["categorical_values_filled"] = int(
        df[categorical_columns].isna().sum().sum()
    )

    for column in categorical_columns:
        df[column] = df[column].fillna("Unknown")

    # 8. Apply the instructor's documented missing-value assumption.
    # This fills missing cells, not an absent source column.
    for column in ["discount", "profit"]:
        summary[f"missing_{column}_filled_with_zero"] = int(
            df[column].isna().sum()
        )
        df[column] = df[column].fillna(0)

    # 9. Interpret discounts using the source's documented format.
    if DISCOUNT_FORMAT == "percent":
        df["discount"] = df["discount"] / 100
    elif DISCOUNT_FORMAT != "rate":
        raise ValueError(
            "DISCOUNT_FORMAT must be 'rate' or 'percent'."
        )

    # Reject invalid rates rather than silently clipping them.
    invalid_discount = ~df["discount"].between(0, 1)

    summary["invalid_discount_rows_removed"] = int(
        invalid_discount.sum()
    )

    df = df.loc[~invalid_discount].copy()

    # 10. Normalize category and region casing.
    for column in ["category", "region"]:
        df[column] = df[column].str.title()

    # Match known labels without damaging names such as PayPal.
    payment_labels = {
        "google pay": "Google Pay",
        "credit card": "Credit Card",
        "debit card": "Debit Card",
        "apple pay": "Apple Pay",
        "cash on delivery": "Cash on Delivery",
        "paypal": "PayPal",
        "bank transfer": "Bank Transfer",
        "unknown": "Unknown",
    }

    status_labels = {
        "delivered": "Delivered",
        "processing": "Processing",
        "returned": "Returned",
        "cancelled": "Cancelled",
        "canceled": "Cancelled",
        "unknown": "Unknown",
    }

    for column, mapping in [
        ("payment_method", payment_labels),
        ("order_status", status_labels),
    ]:
        normalized = df[column].str.casefold()
        df[column] = normalized.map(mapping).fillna(df[column])

    # 11. Remove exact duplicates created by standardization.
    summary["duplicates_after_standardization_removed"] = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates().reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid records remain after cleaning. "
            "Review the source data and cleaning rules."
        )

    summary["output_rows"] = len(df)
    summary["total_rows_removed"] = (
        summary["input_rows"] - summary["output_rows"]
    )

    # Keep the audit summary available for the final report.
    df.attrs["cleaning_summary"] = summary

    print("\n--- CLEANING SUMMARY ---")
    for name, value in summary.items():
        print(f"{name.replace('_', ' ').capitalize()}: {value:,}")

    print("\nRequired-column missing values after cleaning:")
    print(df[REQUIRED_COLUMNS].isna().sum().to_string())

    print("\nKey data types after cleaning:")
    print(
        df[
            ["order_date", "sales", "quantity", "discount", "profit"]
        ].dtypes.to_string()
    )

    print(
        "\nNegative-profit rows retained:",
        int((df["profit"] < 0).sum()),
    )

    return df


if __name__ == "__main__":
    raw_data = load_data()
    cleaned_data = clean_data(raw_data)

    print(
        f"\nDiscount range: "
        f"{cleaned_data['discount'].min():.0%} to "
        f"{cleaned_data['discount'].max():.0%}"
    )

    print("\nOrder statuses after cleaning:")
    print(cleaned_data["order_status"].value_counts().to_string())

    print("\nData cleaning check passed.")
    print("The raw CSV has not been modified.")