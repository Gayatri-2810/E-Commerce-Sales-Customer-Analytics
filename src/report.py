from xml.sax.saxutils import escape

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)

from src.config import (
    PDF_FILE,
    GRAPH_DIR,
    CHART_FILENAMES,
    PROJECT_TITLE,
    DATASET_DESCRIPTION,
    PROJECT_ASSUMPTIONS,
)


CHART_NOTES = {
    "revenue_by_region.png": (
        "Revenue by Region",
        "Compare recorded revenue across regions. Revenue differences "
        "may reflect order counts, product mix, prices, or quantities; "
        "this chart alone does not explain their causes.",
    ),
    "revenue_by_category.png": (
        "Revenue by Category",
        "Compare category contributions to recorded revenue. "
        "High revenue does not necessarily imply a high profit margin.",
    ),
    "profit_by_category.png": (
        "Profit by Category",
        "Compare total profit contributions. Assess these alongside "
        "category revenue and profit margins.",
    ),
    "monthly_revenue_trend.png": (
        "Monthly Revenue Trend",
        "Track recorded revenue by order month. Check monthly order "
        "counts and dataset coverage before interpreting increases "
        "or declines as changes in business performance.",
    ),
    "top_customers.png": (
        "Top 10 Customers",
        "Customers are ranked by recorded revenue and identified by ID. "
        "This ranking does not measure customer lifetime value or "
        "customer profitability.",
    ),
    "discount_vs_profit.png": (
        "Discount vs Profit",
        "Each point represents a transaction row. Negative values "
        "indicate recorded losses. Product mix, transaction size, "
        "and cost differences can affect the observed relationship.",
    ),
    "revenue_distribution.png": (
        "Revenue Distribution",
        "The histogram shows transaction size and skewness. Large "
        "transactions are not automatically errors and should not "
        "be removed solely because they are unusual.",
    ),
    "profit_margin_boxplot.png": (
        "Profit Margin by Category",
        "Boxes show the middle 50% of row margins and the median. "
        "Points beyond the whiskers are potential outliers. "
        "Row margin distributions differ from aggregate margins.",
    ),
    "correlation_heatmap.png": (
        "Numeric Feature Correlations",
        "Pearson correlation summarizes linear association. "
        "Correlation does not establish causation. Revenue, profit, "
        "and profit margin also share mathematical relationships.",
    ),
    "payment_method_usage.png": (
        "Payment Method Usage",
        "Usage counts distinct orders using each method. Differences "
        "describe this synthetic dataset and should not be treated "
        "as evidence of broader consumer preferences.",
    ),
}


def make_paragraph(text, style):
    """Escape dataset text before passing it to ReportLab."""
    return Paragraph(escape(str(text)), style)


def make_table(headers, rows, widths, styles):
    """Create a wrapped table with a repeating header."""
    cells = [
        [make_paragraph(value, styles["TableHeader"]) for value in headers]
    ]

    cells.extend(
        [
            make_paragraph(value, styles["TableBody"])
            for value in row
        ]
        for row in rows
    )

    table = Table(
        cells,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163A5F")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            colors.HexColor("#F1F5F9"),
        ]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#163A5F")),
    ]))

    return table


def draw_page_footer(canvas, doc):
    """Add a page number and report footer."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))

    canvas.drawString(
        doc.leftMargin,
        28,
        "E-Commerce EDA | Synthetic learning data",
    )

    canvas.drawRightString(
        A4[0] - doc.rightMargin,
        28,
        f"Page {doc.page}",
    )

    canvas.restoreState()


def format_number(value, decimals=2):
    """Format missing and valid numeric results."""
    if pd.isna(value):
        return "Undefined"
    return f"{value:,.{decimals}f}"


def generate_pdf_report(df, summaries):
    """Build the complete report from calculated results and charts."""
    missing_charts = [
        filename
        for filename in CHART_FILENAMES
        if not (GRAPH_DIR / filename).is_file()
    ]

    if missing_charts:
        raise FileNotFoundError(
            "Required charts missing: "
            + ", ".join(missing_charts)
            + ". Run the visualization module first."
        )

    PDF_FILE.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(PDF_FILE),
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=42,
        bottomMargin=48,
        title=PROJECT_TITLE,
        author="Gayatri Deshmukh",
    )

    styles = getSampleStyleSheet()

    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 15
    styles["BodyText"].spaceAfter = 8

    styles.add(ParagraphStyle(
        name="TableBody",
        fontName="Helvetica",
        fontSize=9,
        leading=12,
    ))

    styles.add(ParagraphStyle(
        name="TableHeader",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.white,
    ))

    story = []
    kpis = summaries["kpis"]

    def heading(text):
        story.append(make_paragraph(text, styles["Heading1"]))

    def paragraph(text):
        story.append(make_paragraph(text, styles["BodyText"]))

    # 1. Project overview and dataset
    story.append(make_paragraph(PROJECT_TITLE, styles["Title"]))
    paragraph("Complete Exploratory Data Analysis Report")
    paragraph("Prepared by: Gayatri Deshmukh")

    heading("Project Overview")
    paragraph(
        "Analyze sales, profitability, customers, regions, discounts, "
        "payment methods, and order statuses using a reproducible "
        "Python data analysis pipeline."
    )

    heading("Dataset")
    paragraph(DATASET_DESCRIPTION)

    paragraph(
        "Source: https://www.kaggle.com/datasets/"
        "abdelfattahibrahim/global-e-commerce-sales-dataset-20212024"
    )

    paragraph(
        f"Clean records: {len(df):,}. "
        f"Date range: {df['order_date'].min():%d %B %Y} to "
        f"{df['order_date'].max():%d %B %Y}. "
        f"Products: {df['product'].nunique():,}. "
        f"Categories: {df['category'].nunique():,}. "
        f"Regions: {df['region'].nunique():,}."
    )

    paragraph(f"Financial analysis scope: {kpis['scope']}.")
    paragraph(
        "Financial amounts are shown in source currency units. "
        "Customer names are unavailable; customer IDs are used."
    )

    heading("Tools and Workflow")
    paragraph(
        "Python, Pandas, NumPy, Matplotlib, Seaborn, and ReportLab. "
        "Workflow: raw CSV → inspection → cleaning → transformation "
        "→ analysis → charts → business findings → PDF report."
    )

    # 2. Cleaning summary
    story.append(PageBreak())
    heading("Data Cleaning Summary")

    cleaning_summary = df.attrs.get("cleaning_summary", {})

    if cleaning_summary:
        rows = [
            (
                key.replace("_", " ").capitalize(),
                f"{value:,}",
            )
            for key, value in cleaning_summary.items()
        ]

        story.append(make_table(
            ["Cleaning check", "Count"],
            rows,
            [doc.width * 0.75, doc.width * 0.25],
            styles,
        ))
    else:
        paragraph(
            "The cleaning audit was not supplied to this report. "
            "Run the report from the full pipeline to include it."
        )

    story.append(Spacer(1, 12))

    paragraph(
        "Cleaning trims text, parses dates, converts numeric fields, "
        "removes exact duplicates and invalid essential transactions, "
        "and validates discount rates. Negative profits and valid "
        "outliers are retained."
    )

    paragraph(
        "Missing profit or discount cells are filled with zero only "
        "under the documented instructor assumption. Refer to the "
        "audit counts above to determine whether this affected results."
    )

    # 3. KPI summary
    story.append(PageBreak())
    heading("Business KPI Summary")

    margin = format_number(kpis["profit_margin_pct"])
    if pd.notna(kpis["profit_margin_pct"]):
        margin += "%"

    rows = [
        ("Recorded revenue", format_number(kpis["total_revenue"])),
        ("Recorded profit", format_number(kpis["total_profit"])),
        ("Unique orders analyzed", f"{kpis['unique_orders']:,}"),
        ("Known customers analyzed", f"{kpis['unique_customers']:,}"),
        ("Average order value", format_number(kpis["aov"])),
        ("Overall profit margin", margin),
        ("Cancelled orders, all records", f"{kpis['cancelled']:,}"),
        ("Returned orders, all records", f"{kpis['returned']:,}"),
    ]

    story.append(make_table(
        ["Metric", "Value"],
        rows,
        [doc.width * 0.65, doc.width * 0.35],
        styles,
    ))

    story.append(Spacer(1, 12))

    paragraph(
        "AOV = total recorded revenue divided by distinct analyzed "
        "order IDs. Overall profit margin = total recorded profit "
        "divided by total recorded revenue, multiplied by 100."
    )

    paragraph(
        "Status counts cover all clean records. Financial totals follow "
        "the stated analysis scope. When all statuses are included, "
        "these totals are not delivered-only or refund-adjusted results."
    )

    # 4. Findings calculated from the summaries
    heading("Key Findings")

    for name, label_column, metric, description in [
        ("region_revenue", "region", "revenue", "Highest-revenue region"),
        (
            "category_revenue",
            "category",
            "revenue",
            "Highest-revenue category",
        ),
        (
            "category_profit",
            "category",
            "profit",
            "Highest total-profit category",
        ),
        ("product_sales", "product", "revenue", "Highest-revenue product"),
        (
            "top_customers",
            "customer_id",
            "revenue",
            "Highest-revenue customer",
        ),
    ]:
        table = summaries[name]
        if not table.empty:
            row = table.iloc[0]
            paragraph(
                f"{description}: {row[label_column]} "
                f"({format_number(row[metric])})."
            )

    monthly = summaries["monthly_revenue"]
    peak = monthly.loc[monthly["revenue"].idxmax()]

    paragraph(
        f"Highest recorded monthly revenue: {peak['year_month']} "
        f"({format_number(peak['revenue'])})."
    )

    payment = summaries["payment_method"].iloc[0]

    paragraph(
        f"Most-used payment method: {payment['payment_method']} "
        f"({int(payment['unique_orders']):,} distinct orders)."
    )

    correlation = summaries["correlation"].loc[
        "discount", "profit_margin_pct"
    ]

    paragraph(
        "Discount and row profit-margin correlation: "
        f"{format_number(correlation)}. "
        "This association does not establish causation."
    )

    # 5. All ten charts, each on its own page
    for filename in CHART_FILENAMES:
        story.append(PageBreak())

        title, note = CHART_NOTES[filename]
        heading(title)

        image = Image(str(GRAPH_DIR / filename))

        # Preserve the aspect ratio and leave room for captions.
        scale = min(
            doc.width / image.imageWidth,
            440 / image.imageHeight,
        )

        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale
        image.hAlign = "CENTER"

        story.append(image)
        story.append(Spacer(1, 14))
        paragraph(note)
        paragraph(f"Analysis scope: {kpis['scope']}.")

    # 6. Weak-performing products
    story.append(PageBreak())
    heading("Products with the Lowest Total Profit")

    rows = [
        (
            row["product"],
            format_number(row["revenue"]),
            format_number(row["profit"]),
        )
        for _, row in summaries["weak_products"].iterrows()
    ]

    story.append(make_table(
        ["Product", "Revenue", "Profit"],
        rows,
        [doc.width * 0.52, doc.width * 0.24, doc.width * 0.24],
        styles,
    ))

    story.append(Spacer(1, 12))

    paragraph(
        "Lowest total profit does not necessarily mean a product is "
        "loss-making. Low sales volume can also produce a low profit "
        "contribution. Compare margins, volumes, and costs before "
        "deciding on an action."
    )

    heading("Recommendations for Investigation")
    for text in [
        "Review costs and discount practices for low-profit products.",
        "Compare discounts within similar products and categories.",
        "Investigate return and cancellation reasons before proposing fixes.",
        "Check monthly order counts and source coverage before "
        "explaining the late-period revenue decline.",
    ]:
        paragraph(text)

    # 7. Assumptions, limitations, and conclusion
    story.append(PageBreak())
    heading("Assumptions and Limitations")

    for text in PROJECT_ASSUMPTIONS:
        paragraph(text)

    paragraph(
        "Synthetic data supports demonstration of analytical methods; "
        "its patterns are not evidence about an actual business."
    )

    paragraph(
        "Profit is the source's Revenue minus Cost measure. The "
        "available data does not establish that it includes every "
        "operating expense, tax, or refund."
    )

    paragraph(
        "Unknown customer IDs are excluded from customer rankings. "
        "Zero-revenue profit margins are undefined."
    )

    paragraph(
        "Payment counts may overlap if an order uses multiple methods. "
        "Status counts may overlap if an order has multiple statuses."
    )

    paragraph(
        "Observed financial correlations partly reflect shared "
        "calculation formulas. No causal claims are made."
    )

    heading("Conclusion")
    paragraph(
        "The project provides a reproducible view of recorded revenue, "
        "profitability, customer contribution, and transaction patterns. "
        "The findings identify areas for investigation while preserving "
        "the dataset's reporting scope and limitations."
    )

    try:
        doc.build(
            story,
            onFirstPage=draw_page_footer,
            onLaterPages=draw_page_footer,
        )
    except PermissionError as error:
        raise PermissionError(
            "Close the existing PDF in your PDF viewer and run again."
        ) from error

    print("\nPDF report created:", PDF_FILE)


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.data_cleaning import clean_data
    from src.data_transformation import transform_data
    from src.analysis import analyze_data
    from src.visualization import create_visualizations

    raw_data = load_data()
    cleaned_data = clean_data(raw_data)
    transformed_data = transform_data(cleaned_data)
    summaries = analyze_data(transformed_data)

    # Regenerate charts so they match the current results.
    create_visualizations(transformed_data, summaries)
    generate_pdf_report(transformed_data, summaries)

    print("PDF generation completed. Open the PDF to check its layout.")