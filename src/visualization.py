import matplotlib

# Save charts without requiring a separate chart window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import seaborn as sns

from src.config import (
    GRAPH_DIR,
    CHART_FILENAMES,
    INCLUDE_ALL_ORDER_STATUSES,
)
from src.data_loader import load_data
from src.data_cleaning import clean_data
from src.data_transformation import transform_data
from src.analysis import analyze_data


BLUE = "#2563EB"
TEAL = "#0D9488"
RED = "#DC2626"


def compact_number(value, position):
    """Format large axis values using K and M."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.0f}K"
    return f"{value:,.0f}"


def save_chart(fig, filename, title, scope):
    """Apply consistent presentation and save a chart."""
    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
        x=0.08,
        ha="left",
    )

    fig.text(
        0.08,
        0.015,
        f"Synthetic learning data | {scope} | "
        "Financial amounts in source currency units",
        fontsize=8,
        color="#475569",
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.93])

    path = GRAPH_DIR / filename
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)
    print("Saved:", filename)


def create_visualizations(df, summaries):
    """Create the ten required EDA charts."""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    if INCLUDE_ALL_ORDER_STATUSES:
        chart_data = df.copy()
        scope = "All order statuses"
    else:
        chart_data = df.loc[
            df["order_status"].eq("Delivered")
        ].copy()
        scope = "Delivered orders only"

    if chart_data.empty:
        raise ValueError("No records available for visualization.")

    if summaries["kpis"]["scope"] != scope:
        raise ValueError(
            "Chart scope and analysis scope do not match. "
            "Recalculate the summaries."
        )

    money_formatter = FuncFormatter(compact_number)

    # 1. Revenue by region
    table = summaries["region_revenue"].sort_values("revenue")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(table["region"], table["revenue"], color=BLUE)
    ax.set_xlabel("Recorded revenue")
    ax.set_ylabel("Region")
    ax.xaxis.set_major_formatter(money_formatter)
    ax.grid(axis="y", visible=False)

    save_chart(
        fig, "revenue_by_region.png",
        "Revenue by Region", scope,
    )

    # 2. Revenue by category
    table = summaries["category_revenue"].sort_values("revenue")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(table["category"], table["revenue"], color=TEAL)
    ax.set_xlabel("Recorded revenue")
    ax.set_ylabel("Category")
    ax.xaxis.set_major_formatter(money_formatter)
    ax.grid(axis="y", visible=False)

    save_chart(
        fig, "revenue_by_category.png",
        "Revenue by Category", scope,
    )

    # 3. Profit by category
    table = summaries["category_profit"].sort_values("profit")
    colors = [
        TEAL if value >= 0 else RED
        for value in table["profit"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(table["category"], table["profit"], color=colors)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_xlabel("Total recorded profit")
    ax.set_ylabel("Category")
    ax.xaxis.set_major_formatter(money_formatter)
    ax.grid(axis="y", visible=False)

    save_chart(
        fig, "profit_by_category.png",
        "Profit by Category", scope,
    )

    # 4. Monthly revenue trend
    table = summaries["monthly_revenue"].sort_values("year_month")

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(table))

    ax.plot(
        positions,
        table["revenue"],
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=4,
    )

    # Show at most about 12 date labels.
    step = max(1, int(np.ceil(len(table) / 12)))
    ticks = positions[::step]

    ax.set_xticks(ticks)
    ax.set_xticklabels(
        table["year_month"].iloc[::step].astype(str),
        rotation=45,
        ha="right",
    )

    ax.set_xlabel("Order month")
    ax.set_ylabel("Recorded revenue")
    ax.yaxis.set_major_formatter(money_formatter)

    save_chart(
        fig, "monthly_revenue_trend.png",
        "Monthly Revenue Trend", scope,
    )

    # 5. Top ten customers
    table = summaries["top_customers"].sort_values("revenue")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(table["customer_id"], table["revenue"], color=BLUE)
    ax.set_xlabel("Recorded revenue")
    ax.set_ylabel("Customer ID")
    ax.xaxis.set_major_formatter(money_formatter)
    ax.grid(axis="y", visible=False)

    save_chart(
        fig, "top_customers.png",
        "Top 10 Customers by Revenue", scope,
    )

    # 6. Discount versus profit
    fig, ax = plt.subplots(figsize=(11, 7))

    sns.scatterplot(
        data=chart_data,
        x="discount",
        y="profit",
        hue="category",
        alpha=0.35,
        s=22,
        linewidth=0,
        ax=ax,
    )

    ax.axhline(0, color="#334155", linestyle="--", linewidth=1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(money_formatter)
    ax.set_xlabel("Discount rate")
    ax.set_ylabel("Profit per transaction row")
    ax.legend(
        title="Category",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    save_chart(
        fig, "discount_vs_profit.png",
        "Discount vs Profit", scope,
    )

    # 7. Revenue distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        data=chart_data,
        x="revenue",
        bins=40,
        color=BLUE,
        ax=ax,
    )

    ax.set_xlabel("Recorded revenue per transaction row")
    ax.set_ylabel("Transaction rows")
    ax.xaxis.set_major_formatter(money_formatter)

    save_chart(
        fig, "revenue_distribution.png",
        "Revenue Distribution", scope,
    )

    # 8. Profit margin boxplot
    margin_data = chart_data.dropna(subset=["profit_margin_pct"])
    category_order = sorted(chart_data["category"].unique())

    fig, ax = plt.subplots(figsize=(11, 7))

    if margin_data.empty:
        ax.text(
            0.5, 0.5,
            "Profit margin is undefined for all available records.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        sns.boxplot(
            data=margin_data,
            x="profit_margin_pct",
            y="category",
            order=category_order,
            color="#99D8D0",
            fliersize=3,
            ax=ax,
        )

    ax.axvline(0, color="#334155", linestyle="--", linewidth=1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_xlabel("Profit margin")
    ax.set_ylabel("Category")

    save_chart(
        fig, "profit_margin_boxplot.png",
        "Profit Margin Distribution by Category", scope,
    )

    # 9. Correlation heatmap
    # Exclude sales because it is identical to revenue.
    correlation = summaries["correlation"].copy()
    correlation.index = [
        label.replace("_", " ").title()
        for label in correlation.index
    ]
    correlation.columns = correlation.index

    fig, ax = plt.subplots(figsize=(9, 8))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
        ax=ax,
    )

    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)

    save_chart(
        fig, "correlation_heatmap.png",
        "Numeric Feature Correlations", scope,
    )

    # 10. Payment method usage
    table = summaries["payment_method"].sort_values("unique_orders")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        table["payment_method"],
        table["unique_orders"],
        color=TEAL,
    )
    ax.set_xlabel("Distinct orders using each method")
    ax.set_ylabel("Payment method")
    ax.grid(axis="y", visible=False)

    save_chart(
        fig, "payment_method_usage.png",
        "Payment Method Usage", scope,
    )

    # Verify that every expected image exists and contains bytes.
    missing_files = [
        filename
        for filename in CHART_FILENAMES
        if not (GRAPH_DIR / filename).is_file()
        or (GRAPH_DIR / filename).stat().st_size == 0
    ]

    if missing_files:
        raise RuntimeError(
            "Charts missing or empty: " + ", ".join(missing_files)
        )

    print(f"\nAll {len(CHART_FILENAMES)} chart files are present.")
    print("Chart folder:", GRAPH_DIR)


if __name__ == "__main__":
    raw_data = load_data()
    cleaned_data = clean_data(raw_data)
    transformed_data = transform_data(cleaned_data)
    summaries = analyze_data(transformed_data)

    create_visualizations(transformed_data, summaries)

    print("\nVisualization generation check passed.")