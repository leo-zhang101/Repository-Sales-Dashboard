import numpy as np
import pandas as pd


def build_dataset(n_rows: int = 240_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    regions = ["North", "South", "East", "West", "Central"]
    categories = ["Electronics", "Fashion", "Home", "Beauty", "Sports", "Grocery"]

    # Category-level baseline behavior
    category_base_price = {
        "Electronics": 420.0,
        "Fashion": 80.0,
        "Home": 150.0,
        "Beauty": 45.0,
        "Sports": 120.0,
        "Grocery": 25.0,
    }
    category_cost_ratio = {
        "Electronics": 0.76,
        "Fashion": 0.58,
        "Home": 0.63,
        "Beauty": 0.49,
        "Sports": 0.60,
        "Grocery": 0.72,
    }

    # Top products per category (drives realistic concentration)
    top_products = {
        "Electronics": ["E-Flagship-Phone", "E-Ultrabook", "E-Gaming-Laptop", "E-ANC-Headset"],
        "Fashion": ["F-Urban-Sneaker", "F-Denim-Classic", "F-Winter-Jacket", "F-Athleisure-Set"],
        "Home": ["H-Air-Fryer", "H-Ergo-Chair", "H-Robot-Vacuum", "H-Nonstick-Set"],
        "Beauty": ["B-Serum-Pro", "B-SPF50", "B-Repair-Mask", "B-Matte-Foundation"],
        "Sports": ["S-Smartwatch", "S-Trail-Shoes", "S-Yoga-Mat", "S-HIIT-Kit"],
        "Grocery": ["G-Protein-Pack", "G-Coffee-Beans", "G-Organic-Oats", "G-Snack-Box"],
    }

    # Expand long-tail product catalog
    all_products = {}
    for c in categories:
        tops = top_products[c]
        long_tail = [f"{c[:1]}-Product-{i:03d}" for i in range(1, 37)]
        all_products[c] = tops + long_tail

    # Region demand multipliers
    region_multiplier = {
        "North": 1.10,
        "South": 0.92,
        "East": 1.00,
        "West": 1.15,
        "Central": 0.83,
    }

    # Base category mix
    category_probs = np.array([0.22, 0.20, 0.16, 0.14, 0.12, 0.16], dtype=float)
    category_probs /= category_probs.sum()

    # Draw basic dimensions
    order_date = rng.choice(dates.values, size=n_rows, replace=True)
    region = rng.choice(regions, size=n_rows, p=[0.23, 0.18, 0.22, 0.24, 0.13])
    category = rng.choice(categories, size=n_rows, p=category_probs)

    # Build product choice with concentration
    product = []
    for c in category:
        products = all_products[c]
        p = np.ones(len(products), dtype=float)
        p[:4] = np.array([12.0, 8.0, 6.0, 5.0])  # top products dominate
        p[4:] = 1.0
        p /= p.sum()
        product.append(rng.choice(products, p=p))
    product = np.array(product)

    # Seasonality factors by month
    month = pd.to_datetime(order_date).month
    seasonality = np.where(np.isin(month, [11, 12]), 1.30, 1.0)  # holiday peak
    seasonality = np.where(np.isin(month, [6, 7]), seasonality * 1.08, seasonality)
    seasonality = np.where(np.isin(month, [2]), seasonality * 0.92, seasonality)

    # Units sold with mild variance
    units = np.maximum(1, rng.poisson(lam=2.6, size=n_rows))

    # Unit price around category base with noise and seasonality
    base_price = np.array([category_base_price[c] for c in category], dtype=float)
    unit_price = base_price * rng.lognormal(mean=0.0, sigma=0.22, size=n_rows) * seasonality
    unit_price = np.clip(unit_price, 5, None)

    # Discount and region effect
    discount_rate = np.clip(rng.normal(loc=0.09, scale=0.05, size=n_rows), 0.0, 0.35)
    region_effect = np.array([region_multiplier[r] for r in region], dtype=float)

    gross_sales = units * unit_price * region_effect
    net_revenue = gross_sales * (1.0 - discount_rate)

    # Cost and profit
    cost_ratio = np.array([category_cost_ratio[c] for c in category], dtype=float)
    # Higher discounts slightly reduce margin
    variable_cost = net_revenue * (cost_ratio + 0.05 * discount_rate)
    profit = net_revenue - variable_cost

    # Order IDs (simulate multiple lines per order)
    order_sequence = rng.integers(80_000, 190_000, size=n_rows)
    order_id = np.array([f"ORD-{x:06d}" for x in order_sequence])

    # Add calendar fields
    dt = pd.Series(pd.to_datetime(order_date))
    df = pd.DataFrame(
        {
            "order_id": order_id,
            "order_date": dt,
            "year": dt.dt.year,
            "month_num": dt.dt.month,
            "month_name": dt.dt.strftime("%b"),
            "year_month": dt.dt.strftime("%Y-%m"),
            "region": region,
            "category": category,
            "product_name": product,
            "units_sold": units.astype(int),
            "unit_price": unit_price.round(2),
            "discount_rate": discount_rate.round(4),
            "revenue": net_revenue.round(2),
            "profit": profit.round(2),
        }
    )

    return df


def analyze(df: pd.DataFrame) -> dict:
    monthly = (
        df.groupby("year_month", as_index=False)["revenue"]
        .sum()
        .sort_values("year_month")
        .reset_index(drop=True)
    )
    monthly["mom_growth_pct"] = monthly["revenue"].pct_change() * 100

    product_rev = (
        df.groupby("product_name", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    total_revenue = float(df["revenue"].sum())
    product_rev["contribution_pct"] = product_rev["revenue"] / total_revenue * 100
    top10 = product_rev.head(10).copy()
    top10_share = float(top10["revenue"].sum() / total_revenue * 100)

    category_perf = (
        df.groupby("category", as_index=False)
        .agg(revenue=("revenue", "sum"), profit=("profit", "sum"))
        .sort_values("revenue", ascending=False)
    )
    category_perf["margin_pct"] = category_perf["profit"] / category_perf["revenue"] * 100

    region_dist = (
        df.groupby("region", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
    )
    region_dist["share_pct"] = region_dist["revenue"] / total_revenue * 100

    return {
        "monthly": monthly,
        "top10": top10,
        "top10_share_pct": top10_share,
        "category_perf": category_perf,
        "region_dist": region_dist,
        "total_revenue": total_revenue,
        "total_orders": int(df["order_id"].nunique()),
        "avg_order_value": float(total_revenue / df["order_id"].nunique()),
    }


def insight_summary(results: dict) -> str:
    monthly = results["monthly"].copy()
    top10 = results["top10"].copy()
    category_perf = results["category_perf"].copy()
    region_dist = results["region_dist"].copy()

    peak_month = monthly.loc[monthly["revenue"].idxmax(), "year_month"]
    lowest_month = monthly.loc[monthly["revenue"].idxmin(), "year_month"]
    avg_mom = monthly["mom_growth_pct"].dropna().mean()

    top_product = top10.iloc[0]
    best_margin_cat = category_perf.sort_values("margin_pct", ascending=False).iloc[0]
    weakest_margin_cat = category_perf.sort_values("margin_pct", ascending=True).iloc[0]
    top_region = region_dist.iloc[0]

    lines = [
        "# Retail Portfolio Analysis - Business Insights",
        "",
        "## Executive Snapshot",
        f"- Total revenue: {results['total_revenue']:,.2f}",
        f"- Total unique orders: {results['total_orders']:,}",
        f"- Average order value: {results['avg_order_value']:,.2f}",
        "",
        "## Trends and Performance",
        f"- Revenue seasonality is visible, with the highest month in {peak_month} and the lowest month in {lowest_month}.",
        f"- Average month-over-month growth is {avg_mom:.2f}%, indicating a generally stable but seasonal demand pattern.",
        f"- The top-selling product is {top_product['product_name']} with {top_product['contribution_pct']:.2f}% revenue contribution.",
        "",
        "## Portfolio and Concentration Risk",
        f"- Top 10 products contribute {results['top10_share_pct']:.2f}% of total revenue, which indicates moderate concentration risk.",
        "- If one or two leading SKUs underperform, total sales can drop materially during key months.",
        "",
        "## Category and Margin View",
        f"- Best margin category: {best_margin_cat['category']} ({best_margin_cat['margin_pct']:.2f}% margin).",
        f"- Lowest margin category: {weakest_margin_cat['category']} ({weakest_margin_cat['margin_pct']:.2f}% margin).",
        "- Margin gap suggests pricing and discount strategy should be tailored by category, not applied uniformly.",
        "",
        "## Regional Distribution",
        f"- Leading region by revenue: {top_region['region']} ({top_region['share_pct']:.2f}% share).",
        "- Regional mix differences imply targeted inventory and campaign planning can improve sell-through and profitability.",
        "",
        "## Recommended Business Actions",
        "- Diversify demand away from top SKUs by promoting high-potential mid-tier products.",
        "- Create a seasonal operating plan: build inventory before Q4, and run demand stimulation in low months.",
        "- Optimize discount depth by category to protect margin while maintaining conversion.",
        "- Allocate sales resources by regional performance and growth potential, with region-specific product bundles.",
    ]
    return "\n".join(lines)


def main() -> None:
    df = build_dataset()
    results = analyze(df)

    # Power BI ready dataset
    df.to_csv("retail_cleaned_powerbi.csv", index=False)

    # Optional analysis outputs for portfolio artifacts
    results["monthly"].to_csv("analysis_monthly_trend.csv", index=False)
    results["top10"].to_csv("analysis_top10_products.csv", index=False)
    results["category_perf"].to_csv("analysis_category_performance.csv", index=False)
    results["region_dist"].to_csv("analysis_region_distribution.csv", index=False)

    report = insight_summary(results)
    with open("analysis_insights_summary.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("Generated files:")
    print("- retail_cleaned_powerbi.csv")
    print("- analysis_monthly_trend.csv")
    print("- analysis_top10_products.csv")
    print("- analysis_category_performance.csv")
    print("- analysis_region_distribution.csv")
    print("- analysis_insights_summary.md")
    print(f"Rows in cleaned dataset: {len(df):,}")


if __name__ == "__main__":
    main()
