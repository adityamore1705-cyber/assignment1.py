import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("retail_sales.csv")

category_sales = df.groupby(
    'Category'
)['Sales'].sum()

print(category_sales)

category_sales.plot(
    kind='bar',
    title='Category Sales'
)

plt.show()

region_sales = df.groupby(
    'Region'
)['Sales'].sum()

print(region_sales)

monthly_sales = df.groupby(
    'Month'
)['Sales'].sum()

monthly_sales.plot(
    kind='line',
    title='Monthly Sales Trend'
)

plt.show()
