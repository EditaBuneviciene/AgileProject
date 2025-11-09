import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#  Load the dataset
df = pd.read_csv("cleaned_file.csv")

#  Extract all unique series names
series_names = df["Series Name"].unique()
print(" Total unique series:", len(series_names))
series_df=pd.DataFrame(series_names, columns=["Series Name"])
series_df.to_csv("seriesNames.csv", index=False)
print(" List of all available indicators:\n")
for name in series_names:
    print("-", name)

#  Filter for "Life expectancy at birth, total (years)"
life_exp_df = df[df["Series Name"] == "Life expectancy at birth, total (years)"]

#  Convert year columns to numeric
life_exp_years = (
    life_exp_df
    .drop(columns=["Series Name", "Country Name"])
    .apply(pd.to_numeric, errors="coerce")
)

#  Compute correlation between years
corr_matrix = life_exp_years.corr()

# Display correlation matrix
print("\nYear-to-Year Correlation Matrix (Life Expectancy):")
print(corr_matrix)

#  Compute average correlation of each year with all others
avg_corr_per_year = corr_matrix.mean().sort_values(ascending=False)

# Identify strongest and weakest correlation years
strongest_year = avg_corr_per_year.idxmax()
weakest_year = avg_corr_per_year.idxmin()

print(f"\n Strongest average correlation year: {strongest_year} ({avg_corr_per_year.max():.4f}, meaning life expectancy at birth stayed consist compared with other year)")
print(f" Weakest average correlation year: {weakest_year} ({avg_corr_per_year.min():.4f}, meaning most different life expectancy at birth at comparison with other year. )")

#  Plot average correlation per year
"""plt.figure(figsize=(12,6))
plt.plot(avg_corr_per_year.index, avg_corr_per_year.values, marker='o', color='steelblue')
plt.title("Average Correlation of Life Expectancy per Year (2000-2023)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Average correlation with all other years")
plt.grid(True)

# Highlight strongest & weakest years
plt.scatter(strongest_year, avg_corr_per_year.max(), color='green', s=100, label=f'Strongest: {strongest_year}')
plt.scatter(weakest_year, avg_corr_per_year.min(), color='red', s=100, label=f'Weakest: {weakest_year}')"""