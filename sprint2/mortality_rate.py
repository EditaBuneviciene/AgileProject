import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("cleaned_file.csv")

# --- Data preparation ---
mortality_df = df[df["Series Name"].str.contains("mortality rate", case=False, na=False)]
mortality_df = mortality_df[mortality_df["Country Name"].isin(["Ireland", "Pakistan"])]

# Separate male and female
male = mortality_df[mortality_df["Series Name"].str.contains("male", case=False)]
female = mortality_df[mortality_df["Series Name"].str.contains("female", case=False)]

# Convert from wide to long format(adds Year and Moratlity columns automatucly)
male_melted = male.melt(id_vars=["Country Name", "Series Name"], var_name="Year", value_name="Mortality Rate")
female_melted = female.melt(id_vars=["Country Name", "Series Name"], var_name="Year", value_name="Mortality Rate")

# Convert Year to numeric
male_melted["Year"] = pd.to_numeric(male_melted["Year"], errors="coerce")
female_melted["Year"] = pd.to_numeric(female_melted["Year"], errors="coerce")

# Combine male and female
combined = pd.concat([
    male_melted.assign(Gender="Male"),
    female_melted.assign(Gender="Female")
])

# --- Filter for years 2020 to 2022 ---
combined = combined[combined["Year"].between(2020, 2022)]

# --- Create combined Country-Gender column for proper grouping ---
combined["Country-Gender"] = combined["Country Name"] + " - " + combined["Gender"]

# --- Plot ---
sns.set_style("whitegrid")
plt.figure(figsize=(14, 6))

# Grouped barplot using combined column
barplot = sns.barplot(
    data=combined,
    x="Year",
    y="Mortality Rate",
    hue="Country-Gender",
    palette={"Ireland - Male": "royalblue", "Ireland - Female": "lightpink",
             "Pakistan - Male": "navy", "Pakistan - Female": "tomato"},
    ci=None
)

# Add numeric labels above bars
for p in barplot.patches:
    height = p.get_height()
    barplot.annotate(f'{height:.1f}',
                     (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='bottom',
                     fontsize=9, rotation=90)

# Formatting
plt.xticks(rotation=45)
plt.xlabel("Year")
plt.ylabel("Mortality Rate")
plt.title("Mortality Rate (2020-2022) by Gender and Country", fontsize=16)
plt.legend(title="Country & Gender")
plt.tight_layout()
plt.show()
