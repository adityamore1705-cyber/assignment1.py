import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("employee_data.csv")

print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)

df.drop_duplicates(inplace=True)

Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['Salary'] >= lower) &
        (df['Salary'] <= upper)]

sns.histplot(df['Salary'], kde=True)
plt.title("Salary Distribution")
plt.show()

sns.boxplot(x='Department',
            y='Salary',
            data=df)
plt.show()

sns.heatmap(df.corr(numeric_only=True),
            annot=True)

plt.show()
