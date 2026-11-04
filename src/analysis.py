import numpy as np
import pandas as pd

from src.config import (
    ANALYSIS_DIR,
    INSIGHTS_FILE,
    INCLUDE_ALL_ORDER_STATUSES,
    PROJECT_ASSUMPTIONS,
)
from src.data_loader import load_data
from src.data_cleaning import clean_data
from src.data_transformation import transform_data


def summarize_performance(df, column):
    """Summarize revenue, profit, and orders for a business dimension."""
    result = (
        df.groupby(column, dropna=False)
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            unique_orders=("order_id", "nunique"),
            quantity=("quantity", "sum"),
        )
        .reset_index()
    )

    result["profit_margin_pct"] = (
        result["profit"]
        .div(result["revenue"].replace(0, np.nan))
        .mul(100)
    )

    return result.sort_values(
        "revenue", ascending=False
    ).reset_index(drop=True)


def analyze_data(df):
    """Calculate business summaries and save tables and insights."""
    if df.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    # Preserve all records for status counts.
    all_data = df.copy()

    # Follow the configured revenue reporting scope.
    if INCLUDE_ALL_ORDER_STATUSES:
        analysis_data = all_data.copy()
        scope = "All order statuses"
    else:
        analysis_data = all_data.loc[
            all_data["order_status"].eq("Delivered")
        ].copy()
        scope = "Delivered orders only"

    if analysis_data.empty:
        raise ValueError("No records match the selected analysis scope.")

    total_revenue = float(analysis_data["revenue"].sum())
    total_profit = float(analysis_data["profit"].sum())
    unique_orders = int(analysis_data["order_id"].nunique())

    aov = total_revenue / unique_orders if unique_orders else np.nan

    profit_margin = (
        total_profit / total_revenue * 100
        if total_revenue != 0
        else np.nan
    )

    cancelled = int(
        all_data.loc[
            all_data["order_status"].eq("Cancelled"), "order_id"
        ].nunique()
    )

    returned = int(
        all_data.loc[
            all_data["order_status"].eq("Returned"), "order_id"
        ].nunique()
    )

    kpis = {
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "unique_orders": unique_orders,
        "unique_customers": int(
            analysis_data.loc[
                analysis_data["customer_id"].ne("Unknown"),
                "customer_id",
            ].nunique()
        ),
        "aov": aov,
        "profit_margin_pct": profit_margin,
        "cancelled": cancelled,
        "returned": returned,
        "analyzed_rows": len(analysis_data),
        "source_unique_orders": int(all_data["order_id"].nunique()),
        "scope": scope,
    }

    summaries = {
        "kpis": kpis,
        "region_revenue": summarize_performance(
            analysis_data, "region"
        ),
        "category_revenue": summarize_performance(
            analysis_data, "category"
        ),
        "product_sales": summarize_performance(
            analysis_data, "product"
        ),
    }

    summaries["category_profit"] = (
        summaries["category_revenue"]
        .sort_values("profit", ascending=False)
        .reset_index(drop=True)
    )

    # Exclude unidentified customers from customer rankings.
    known_customers = analysis_data.loc[
        analysis_data["customer_id"].ne("Unknown")
    ]

    summaries["top_customers"] = (
        summarize_performance(known_customers, "customer_id")
        .head(10)
        .reset_index(drop=True)
    )

    summaries["monthly_revenue"] = (
        analysis_data.groupby("year_month")
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            unique_orders=("order_id", "nunique"),
        )
        .sort_index()
        .reset_index()
    )

    # Count distinct orders for payment usage, not transaction rows.
    # If an order uses multiple methods, it counts under each method.
    summaries["payment_method"] = (
        analysis_data.groupby("payment_method")
        .agg(unique_orders=("order_id", "nunique"))
        .sort_values("unique_orders", ascending=False)
        .reset_index()
    )

    summaries["status_counts"] = (
        all_data.groupby("order_status")
        .agg(
            order_rows=("order_id", "size"),
            unique_orders=("order_id", "nunique"),
        )
        .sort_values("unique_orders", ascending=False)
        .reset_index()
    )

    discount_summary = (
        analysis_data.groupby(
            "discount_category", observed=True
        )
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            average_profit=("profit", "mean"),
            order_rows=("order_id", "size"),
        )
        .reset_index()
    )

    discount_summary["profit_margin_pct"] = (
        discount_summary["profit"]
        .div(discount_summary["revenue"].replace(0, np.nan))
        .mul(100)
    )

    summaries["discount_profit"] = discount_summary

    summaries["weak_products"] = (
        summaries["product_sales"]
        .sort_values("profit")
        .head(10)
        .reset_index(drop=True)
    )

    summaries["weak_categories"] = (
        summaries["category_revenue"]
        .sort_values("profit")
        .reset_index(drop=True)
    )

    numeric_columns = [
        "revenue",
        "quantity",
        "discount",
        "profit",
        "profit_margin_pct",
    ]

    summaries["numeric_summary"] = (
        analysis_data[numeric_columns].describe().T
    )

    summaries["correlation"] = (
        analysis_data[numeric_columns].corr()
    )

    # Reconcile grouped totals with the headline revenue.
    for name in [
        "region_revenue",
        "category_revenue",
        "product_sales",
        "monthly_revenue",
    ]:
        if not np.isclose(
            summaries[name]["revenue"].sum(),
            total_revenue,
            rtol=0,
            atol=0.01,
        ):
            raise ValueError(
                f"Revenue reconciliation failed for {name}."
            )

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Save every summary table for inspection and reuse.
    for name, table in summaries.items():
        if isinstance(table, pd.DataFrame):
            keep_index = name in ["numeric_summary", "correlation"]
            table.to_csv(
                ANALYSIS_DIR / f"{name}.csv",
                index=keep_index,
                encoding="utf-8-sig",
            )

    pd.DataFrame([kpis]).to_csv(
        ANALYSIS_DIR / "kpis.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_business_insights(analysis_data, summaries)

    print("\n--- BUSINESS KPI SUMMARY ---")
    print(f"Analysis scope: {scope}")
    print(f"Total revenue: {total_revenue:,.2f}")
    print(f"Total profit: {total_profit:,.2f}")
    print(f"Unique orders: {unique_orders:,}")
    print(f"Average order value: {aov:,.2f}")
    print(f"Overall profit margin: {profit_margin:.2f}%")
    print(f"Cancelled orders, all records: {cancelled:,}")
    print(f"Returned orders, all records: {returned:,}")

    print("\nRevenue by region:")
    print(summaries["region_revenue"].to_string(index=False))

    print("\nProfit by category:")
    print(summaries["category_profit"].to_string(index=False))

    print("\nAnalysis tables saved to:", ANALYSIS_DIR)
    print("Business insights saved to:", INSIGHTS_FILE)

    return summaries


def write_business_insights(df, summaries):
    """Write observed findings separately from recommendations."""
    kpis = summaries["kpis"]

    region = summaries["region_revenue"].iloc[0]
    category = summaries["category_profit"].iloc[0]
    product = summaries["product_sales"].iloc[0]
    month = summaries["monthly_revenue"].sort_values(
        "revenue", ascending=False
    ).iloc[0]
    payment = summaries["payment_method"].iloc[0]

    correlation = df["discount"].corr(df["profit"])

    correlation_text = (
        f"{correlation:.3f}"
        if pd.notna(correlation)
        else "undefined because there is insufficient variation or data"
    )

    lines = [
        "E-COMMERCE BUSINESS INSIGHTS",
        "=" * 40,
        "",
        "REPORTING SCOPE",
        kpis["scope"],
        "Dataset: synthetic learning data.",
        "Amounts are reported in source currency units.",
        "",
        "OBSERVED FINDINGS",
        f"Total recorded revenue: {kpis['total_revenue']:,.2f}",
        f"Total recorded profit: {kpis['total_profit']:,.2f}",
        f"Unique orders analyzed: {kpis['unique_orders']:,}",
        f"Average order value: {kpis['aov']:,.2f}",
        f"Overall profit margin: {kpis['profit_margin_pct']:.2f}%",
        (
            f"Highest-revenue region: {region['region']} "
            f"({region['revenue']:,.2f})."
        ),
        (
            f"Highest total-profit category: {category['category']} "
            f"({category['profit']:,.2f})."
        ),
        (
            f"Highest-revenue product: {product['product']} "
            f"({product['revenue']:,.2f})."
        ),
        (
            f"Highest recorded monthly revenue: {month['year_month']} "
            f"({month['revenue']:,.2f})."
        ),
        (
            f"Most-used payment method by distinct orders: "
            f"{payment['payment_method']} "
            f"({int(payment['unique_orders']):,} orders)."
        ),
        f"Cancelled orders across all records: {kpis['cancelled']:,}",
        f"Returned orders across all records: {kpis['returned']:,}",
        (
            "Pearson correlation between row discount and row profit: "
            f"{correlation_text}."
        ),
        "Correlation does not establish causation.",
        "",
        "TOP CUSTOMERS BY REVENUE",
        summaries["top_customers"].to_string(index=False),
        "",
        "LOWEST TOTAL-PROFIT PRODUCTS",
        summaries["weak_products"].to_string(index=False),
        "",
        "RECOMMENDATIONS FOR INVESTIGATION",
        "Review pricing and costs for the lowest-profit products.",
        "Compare discount bands within similar products or categories.",
        "Investigate cancellation and return reasons before proposing fixes.",
        "Check date coverage before interpreting monthly changes.",
        "",
        "ASSUMPTIONS AND LIMITATIONS",
        *PROJECT_ASSUMPTIONS,
        (
            "Lowest total profit does not necessarily mean a product "
            "has a negative profit or the lowest profit margin."
        ),
        (
            "Status counts cover all records. A multi-status order "
            "can appear in more than one status group."
        ),
        (
            "Payment usage counts orders using each method; "
            "multi-method orders may appear in multiple groups."
        ),
    ]

    INSIGHTS_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raw_data = load_data()
    cleaned_data = clean_data(raw_data)
    transformed_data = transform_data(cleaned_data)
    results = analyze_data(transformed_data)

    print("\nBusiness analysis check passed.")