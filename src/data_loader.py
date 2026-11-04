import pandas as pd

from src.config import RAW_FILE, COLUMN_ALIASES, REQUIRED_COLUMNS


def standardize_columns(df):
    """Convert source headers into consistent project column names."""
    df = df.copy()
    renamed_columns = {}

    for column in df.columns:
        normalized = str(column).strip().lower().replace("_", " ")
        normalized = " ".join(normalized.split())

        renamed_columns[column] = COLUMN_ALIASES.get(
            normalized,
            normalized.replace(" ", "_"),
        )

    df = df.rename(columns=renamed_columns)

    # Different source headers must not become the same project column.
    duplicates = df.columns[df.columns.duplicated()].tolist()

    if duplicates:
        raise ValueError(
            f"Column mapping produced duplicate names: {duplicates}"
        )

    return df


def inspect_data(df):
    """Display the dataset's structure and initial quality checks."""
    print("\n--- DATASET OVERVIEW ---")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print(f"\nExact duplicate rows: {df.duplicated().sum():,}")
    print(f"Unique orders: {df['order_id'].nunique():,}")
    print(f"Unique customers: {df['customer_id'].nunique():,}")

    print("\nOrder status counts:")
    print(df["order_status"].value_counts(dropna=False).to_string())

    print("\nPayment method counts:")
    print(df["payment_method"].value_counts(dropna=False).to_string())


def load_data():
    """Load, standardize, validate, and inspect the raw CSV."""
    if not RAW_FILE.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}\n"
            "Place ecommerce_raw.csv inside data/raw."
        )

    try:
        # Preserve identifiers as text, including any leading zeros.
        df = pd.read_csv(
            RAW_FILE,
            encoding="utf-8-sig",
            dtype={
                "Order_ID": "string",
                "Customer_ID": "string",
            },
        )
    except pd.errors.EmptyDataError as error:
        raise ValueError("The CSV file contains no readable data.") from error
    except pd.errors.ParserError as error:
        raise ValueError(
            "The CSV could not be parsed. Check its delimiter and row structure."
        ) from error

    if df.empty:
        raise ValueError("The CSV contains headers but no data rows.")

    df = standardize_columns(df)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns missing: "
            + ", ".join(missing_columns)
            + ". Check the CSV headers and COLUMN_ALIASES in config.py."
        )

    inspect_data(df)

    return df


if __name__ == "__main__":
    dataset = load_data()

    print("\nData loading check passed.")
    print("The raw file has not been modified.")