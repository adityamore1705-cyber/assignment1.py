import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


file_name = "retail_sales.csv"

try:
    df = pd.read_csv(file_name)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: '{file_name}' was not found.")
    print("Place the CSV file in the same folder as this program.")
    exit()

print("\nFirst five rows:")
print(df.head())

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# ------------------------------------------------
# 3. CLEAN COLUMN NAMES
# ------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nCleaned column names:")
print(df.columns.tolist())

# ------------------------------------------------
# 4. CHECK MISSING VALUES
# ------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())

# ------------------------------------------------
# 5. REMOVE DUPLICATES
# ------------------------------------------------

duplicates = df.duplicated().sum()

print(f"\nDuplicate rows: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")

# ------------------------------------------------
# 6. CONVERT DATE COLUMN
# ------------------------------------------------

if "date" in df.columns:

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Remove rows with invalid dates
    df = df.dropna(subset=["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%B")

# ------------------------------------------------
# 7. CREATE REVENUE COLUMN
# ------------------------------------------------

if "quantity" in df.columns and "price" in df.columns:

    df["revenue"] = df["quantity"] * df["price"]

elif "sales" in df.columns:

    df["revenue"] = df["sales"]

elif "revenue" in df.columns:

    df["revenue"] = pd.to_numeric(
        df["revenue"],
        errors="coerce"
    )

else:
    print(
        "\nWarning: No sales/revenue column was found."
    )

# ------------------------------------------------
# 8. DESCRIPTIVE STATISTICS
# ------------------------------------------------

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe(include="all").transpose())

# ------------------------------------------------
# 9. TOTAL REVENUE
# ------------------------------------------------

if "revenue" in df.columns:

    total_revenue = df["revenue"].sum()
    average_revenue = df["revenue"].mean()

    print("\n" + "=" * 60)
    print("REVENUE ANALYSIS")
    print("=" * 60)

    print(
        f"Total revenue: ${total_revenue:,.2f}"
    )

    print(
        f"Average transaction revenue: "
        f"${average_revenue:,.2f}"
    )

# ------------------------------------------------
# 10. TOP PRODUCTS
# ------------------------------------------------

if "product" in df.columns and "revenue" in df.columns:

    product_sales = (
        df.groupby("product")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 60)
    print("TOP 10 PRODUCTS")
    print("=" * 60)

    print(product_sales.head(10))

    plt.figure(figsize=(10, 6))

    product_sales.head(10).sort_values().plot(
        kind="barh"
    )

    plt.title("Top 10 Products by Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("Product")

    plt.tight_layout()
    plt.show()

# ------------------------------------------------
# 11. CATEGORY ANALYSIS
# ------------------------------------------------

if "category" in df.columns and "revenue" in df.columns:

    category_sales = (
        df.groupby("category")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 60)
    print("SALES BY CATEGORY")
    print("=" * 60)

    print(category_sales)

    plt.figure(figsize=(10, 6))

    category_sales.plot(
        kind="bar"
    )

    plt.title("Revenue by Category")
    plt.xlabel("Category")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

# ------------------------------------------------
# 12. MONTHLY SALES TREND
# ------------------------------------------------

if "date" in df.columns and "revenue" in df.columns:

    monthly_sales = (
        df.groupby(
            df["date"].dt.to_period("M")
        )["revenue"]
        .sum()
    )

    print("\n" + "=" * 60)
    print("MONTHLY SALES")
    print("=" * 60)

    print(monthly_sales)

    plt.figure(figsize=(12, 6))

    monthly_sales.plot(
        marker="o"
    )

    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ------------------------------------------------
# 13. YEARLY SALES
# ------------------------------------------------

if "year" in df.columns and "revenue" in df.columns:

    yearly_sales = (
        df.groupby("year")["revenue"]
        .sum()
    )

    print("\n" + "=" * 60)
    print("YEARLY REVENUE")
    print("=" * 60)

    print(yearly_sales)

    plt.figure(figsize=(10, 6))

    yearly_sales.plot(
        kind="bar"
    )

    plt.title("Revenue by Year")
    plt.xlabel("Year")
    plt.ylabel("Revenue")

    plt.tight_layout()
    plt.show()

# ------------------------------------------------
# 14. CUSTOMER ANALYSIS
# ------------------------------------------------

if "customer_id" in df.columns:

    customers = df["customer_id"].nunique()

    print("\n" + "=" * 60)
    print("CUSTOMER ANALYSIS")
    print("=" * 60)

    print(
        f"Number of unique customers: {customers}"
    )

    if "revenue" in df.columns:

        customer_revenue = (
            df.groupby("customer_id")["revenue"]
            .sum()
            .sort_values(ascending=False)
        )

        print("\nTop 10 customers by revenue:")
        print(customer_revenue.head(10))

# ------------------------------------------------
# 15. CORRELATION ANALYSIS
# ------------------------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns

if len(numeric_columns) >= 2:

    correlation = df[numeric_columns].corr()

    print("\n" + "=" * 60)
    print("CORRELATION MATRIX")
    print("=" * 60)

    print(correlation)

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Between Numerical Variables")

    plt.tight_layout()
    plt.show()

# ------------------------------------------------
# 16. SALES DISTRIBUTION
# ------------------------------------------------

if "revenue" in df.columns:

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["revenue"],
        bins=30,
        kde=True
    )

    plt.title("Revenue Distribution")
    plt.xlabel("Revenue")
    plt.ylabel("Number of Transactions")

    plt.tight_layout()
    plt.show()

# ------------------------------------------------
# 17. OUTLIER ANALYSIS
# ------------------------------------------------

if "revenue" in df.columns:

    Q1 = df["revenue"].quantile(0.25)
    Q3 = df["revenue"].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df["revenue"] < lower_limit) |
        (df["revenue"] > upper_limit)
    ]

    print("\n" + "=" * 60)
    print("OUTLIER ANALYSIS")
    print("=" * 60)

    print(
        f"Number of revenue outliers: {len(outliers)}"
    )

# ------------------------------------------------
# 18. KEY BUSINESS INSIGHTS
# ------------------------------------------------

print("\n" + "=" * 60)
print("KEY BUSINESS INSIGHTS")
print("=" * 60)

if "revenue" in df.columns:

    highest_transaction = df["revenue"].max()

    print(
        f"Highest transaction revenue: "
        f"${highest_transaction:,.2f}"
    )

if "product" in df.columns and "revenue" in df.columns:

    best_product = product_sales.idxmax()

    print(
        f"Best-performing product: {best_product}"
    )

if "category" in df.columns and "revenue" in df.columns:

    best_category = category_sales.idxmax()

    print(
        f"Best-performing category: {best_category}"
    )

if "date" in df.columns and "revenue" in df.columns:

    best_month = monthly_sales.idxmax()

    print(
        f"Highest-revenue month: {best_month}"
    )

print("\nAnalysis completed successfully!")
