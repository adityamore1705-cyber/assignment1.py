import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_name = "data.csv"

try:
    df = pd.read_csv(file_name)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: {file_name} was not found.")
    print("Make sure the CSV file is in the same folder as this Python file.")
    exit()

print("\n" + "=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nDataset shape:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)


print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe(include="all").transpose())


print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Values": missing,
    "Percentage": missing_percent
})

print(missing_table[missing_table["Missing Values"] > 0])

if missing.sum() == 0:
    print("No missing values found.")



print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

print("\n" + "=" * 60)
print("UNIQUE VALUES")
print("=" * 60)

for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")

# --------------------------------------------------
# 7. NUMERICAL AND CATEGORICAL COLUMNS
# --------------------------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\nNumerical columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)

# --------------------------------------------------
# 8. HISTOGRAMS
# --------------------------------------------------

if len(numeric_columns) > 0:

    df[numeric_columns].hist(
        figsize=(14, 10),
        bins=20
    )

    plt.suptitle(
        "Distribution of Numerical Features",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# 9. BOXPLOTS
# --------------------------------------------------

if len(numeric_columns) > 0:

    plt.figure(figsize=(14, 8))

    sns.boxplot(
        data=df[numeric_columns]
    )

    plt.title("Boxplots of Numerical Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# 10. CORRELATION ANALYSIS
# --------------------------------------------------

if len(numeric_columns) >= 2:

    correlation_matrix = df[numeric_columns].corr()

    print("\n" + "=" * 60)
    print("CORRELATION MATRIX")
    print("=" * 60)

    print(correlation_matrix)

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# 11. CATEGORICAL DATA VISUALIZATION
# --------------------------------------------------

for column in categorical_columns:

    # Avoid extremely large category plots
    if df[column].nunique() <= 15:

        plt.figure(figsize=(10, 6))

        order = df[column].value_counts().index

        sns.countplot(
            data=df,
            x=column,
            order=order
        )

        plt.title(f"Distribution of {column}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# --------------------------------------------------
# 12. SCATTER PLOTS
# --------------------------------------------------

if len(numeric_columns) >= 2:

    # Select the first two numerical columns
    x_column = numeric_columns[0]
    y_column = numeric_columns[1]

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x=x_column,
        y=y_column
    )

    plt.title(f"{x_column} vs {y_column}")
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# 13. OUTLIER DETECTION USING IQR
# --------------------------------------------------

print("\n" + "=" * 60)
print("OUTLIER ANALYSIS")
print("=" * 60)

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(
        f"{column}: {len(outliers)} outliers"
    )

# --------------------------------------------------
# 14. TOP CORRELATIONS
# --------------------------------------------------

if len(numeric_columns) >= 2:

    corr_matrix = df[numeric_columns].corr()

    correlation_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):

            column1 = corr_matrix.columns[i]
            column2 = corr_matrix.columns[j]

            correlation = corr_matrix.iloc[i, j]

            correlation_pairs.append(
                (column1, column2, correlation)
            )

    correlation_pairs.sort(
        key=lambda x: abs(x[2]),
        reverse=True
    )

    print("\n" + "=" * 60)
    print("STRONGEST CORRELATIONS")
    print("=" * 60)

    for column1, column2, correlation in correlation_pairs[:10]:

        print(
            f"{column1} <-> {column2}: "
            f"{correlation:.3f}"
        )

# --------------------------------------------------
# 15. AUTOMATIC INSIGHTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

if missing.sum() > 0:
    print(
        f"There are {missing.sum()} missing values "
        "that may require data cleaning."
    )
else:
    print("The dataset contains no missing values.")

if duplicates > 0:
    print(
        f"There are {duplicates} duplicate rows."
    )
else:
    print("No duplicate rows were detected.")

if len(numeric_columns) >= 2:

    strongest = correlation_pairs[0]

    print(
        f"The strongest numerical correlation is between "
        f"{strongest[0]} and {strongest[1]} "
        f"with a correlation of {strongest[2]:.3f}."
    )

print("\nEDA completed successfully!")
