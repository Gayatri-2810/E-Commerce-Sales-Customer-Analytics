from pathlib import Path

# Find the project folder automatically.
# This works even if you later move the project elsewhere.
ROOT_DIR = Path(__file__).resolve().parents[1]

# Input and output paths
RAW_FILE = ROOT_DIR / "data" / "raw" / "ecommerce_raw.csv"
CLEAN_FILE = ROOT_DIR / "data" / "processed" / "ecommerce_cleaned.csv"

GRAPH_DIR = ROOT_DIR / "outputs" / "graphs"
ANALYSIS_DIR = ROOT_DIR / "outputs" / "analysis"
REPORT_DIR = ROOT_DIR / "outputs" / "reports"

INSIGHTS_FILE = ANALYSIS_DIR / "business_insights.txt"
PDF_FILE = REPORT_DIR / "ecommerce_eda_report.pdf"

# Columns needed by this project's analysis.
# Customer names are unavailable: customers are identified by ID.
REQUIRED_COLUMNS = [
    "order_id",
    "order_date",
    "customer_id",
    "product",
    "category",
    "region",
    "sales",
    "quantity",
    "discount",
    "profit",
    "payment_method",
    "order_status",
]

# Keys below match headers after trimming, lowercasing,
# and replacing underscores with spaces.
COLUMN_ALIASES = {
    "order id": "order_id",
    "order date": "order_date",
    "customer id": "customer_id",
    "product name": "product",
    "product": "product",
    "category": "category",
    "sub category": "sub_category",
    "region": "region",
    "country": "country",
    "unit price": "unit_price",
    "quantity": "quantity",
    "discount": "discount",
    "revenue": "sales",
    "sales": "sales",
    "cost": "cost",
    "profit": "profit",
    "profit margin %": "source_profit_margin_pct",
    "shipping cost": "shipping_cost",
    "shipping method": "shipping_method",
    "shipping days": "shipping_days",
    "payment method": "payment_method",
    "order status": "order_status",
    "customer gender": "customer_gender",
    "customer segment": "customer_segment",
    "year": "year",
    "month": "month",
    "quarter": "quarter",
    "season": "season",
}

# Source-specific interpretation
DATE_FORMAT = "%Y-%m-%d"
DISCOUNT_FORMAT = "rate"

# Follow the instructor's default:
# retain all statuses in the main analysis.
INCLUDE_ALL_ORDER_STATUSES = True

# Exact output filenames required by the instructor
CHART_FILENAMES = [
    "revenue_by_region.png",
    "revenue_by_category.png",
    "profit_by_category.png",
    "monthly_revenue_trend.png",
    "top_customers.png",
    "discount_vs_profit.png",
    "revenue_distribution.png",
    "profit_margin_boxplot.png",
    "correlation_heatmap.png",
    "payment_method_usage.png",
]

PROJECT_TITLE = "E-Commerce Sales & Customer Analysis"

DATASET_DESCRIPTION = (
    "Global E-Commerce Sales Dataset, 2021–2024. "
    "Synthetic data for learning and portfolio analysis."
)

PROJECT_ASSUMPTIONS = [
    "The source Revenue column is treated as recorded sales.",
    "Recorded sales already reflect discounts; do not discount them again.",
    "Discount is a rate: 0.10 represents 10%.",
    "Customers are identified by Customer_ID; names are unavailable.",
    "Average order value uses distinct Order_ID values.",
    "Main revenue and profit totals include all order statuses.",
    "These totals are not delivered-only or refund-adjusted results.",
    "Negative profit values are retained for profitability analysis.",
]


def create_output_directories():
    """Create the folders used for generated outputs."""
    for directory in [
        CLEAN_FILE.parent,
        GRAPH_DIR,
        ANALYSIS_DIR,
        REPORT_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def verify_configuration():
    """Check paths and confirm that the raw CSV exists."""
    print("Project folder:", ROOT_DIR)
    print("Raw CSV:", RAW_FILE)

    if not RAW_FILE.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}\n"
            "Copy your CSV into data/raw and name it ecommerce_raw.csv."
        )

    create_output_directories()

    print("Raw CSV found.")
    print("Output folders are ready.")
    print("Configuration check passed.")


if __name__ == "__main__":
    verify_configuration()