import pandas as pd

from src.config import (
    CLEAN_FILE,
    GRAPH_DIR,
    CHART_FILENAMES,
    INSIGHTS_FILE,
    PDF_FILE,
    create_output_directories,
)
from src.data_loader import load_data
from src.data_cleaning import clean_data
from src.data_transformation import transform_data
from src.analysis import analyze_data
from src.visualization import create_visualizations
from src.report import generate_pdf_report


def verify_outputs(expected_rows):
    """Check required outputs and the saved CSV row count."""
    required_files = [
        CLEAN_FILE,
        INSIGHTS_FILE,
        PDF_FILE,
        *[GRAPH_DIR / name for name in CHART_FILENAMES],
    ]

    missing_or_empty = [
        str(path)
        for path in required_files
        if not path.is_file() or path.stat().st_size == 0
    ]

    if missing_or_empty:
        raise RuntimeError(
            "Required outputs are missing or empty:\n"
            + "\n".join(missing_or_empty)
        )

    saved_data = pd.read_csv(
        CLEAN_FILE,
        dtype={
            "order_id": "string",
            "customer_id": "string",
        },
    )

    if len(saved_data) != expected_rows:
        raise ValueError(
            f"Saved CSV has {len(saved_data):,} rows; "
            f"expected {expected_rows:,}."
        )

    print("\nRequired output files are present and non-empty.")
    print(f"Saved CSV row count verified: {len(saved_data):,}")


def main():
    """Execute the complete EDA pipeline."""
    create_output_directories()

    print("\n1/7 — Loading and inspecting raw data")
    raw_data = load_data()

    print("\n2/7 — Cleaning data")
    cleaned_data = clean_data(raw_data)

    print("\n3/7 — Creating analysis columns")
    transformed_data = transform_data(cleaned_data)

    print("\n4/7 — Calculating business metrics and insights")
    summaries = analyze_data(transformed_data)

    print("\n5/7 — Generating the ten charts")
    create_visualizations(transformed_data, summaries)

    print("\n6/7 — Saving the cleaned dataset and PDF report")

    transformed_data.to_csv(
        CLEAN_FILE,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    generate_pdf_report(transformed_data, summaries)

    print("\n7/7 — Verifying generated outputs")
    verify_outputs(expected_rows=len(transformed_data))

    print("\nPROJECT PIPELINE COMPLETED")
    print("Cleaned dataset:", CLEAN_FILE)
    print("Charts:", GRAPH_DIR)
    print("Business insights:", INSIGHTS_FILE)
    print("PDF report:", PDF_FILE)


if __name__ == "__main__":
    main()