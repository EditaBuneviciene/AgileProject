import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("cleaned_file.csv")

# --- Filter for relevant indicators and countries ---
indicators = [
    "Mortality from CVD, cancer, diabetes or CRD between exact ages 30 and 70 (%)",
    "Nurses and midwives (per 1,000 people)"
]
countries = ["Ireland", "China"]

filtered = df[df["Series Name"].isin(indicators) & df["Country Name"].isin(countries)]

# --- Convert from wide to long format ---
melted = filtered.melt(
    id_vars=["Country Name", "Series Name"],
    var_name="Year",
    value_name="Value"
)

# Convert Year to numeric
melted["Year"] = pd.to_numeric(melted["Year"], errors="coerce")
melted = melted[melted["Year"].between(2000, 2021)]
melted = melted.dropna(subset=["Year", "Value"])

# --- Plot setup ---
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# Line plot for both indicators
sns.lineplot(
    data=melted,
    x="Year",
    y="Value",
    hue="Series Name",
    style="Country Name",
    markers=True,
    dashes=False,
    palette="tab10"
)

# --- Formatting ---
plt.title("Mortality (CVD, Cancer, Diabetes, CRD) vs Nurses and midwives (per 1,000 people)", fontsize=15)
plt.xlabel("Year")
plt.ylabel("Percentage (%)")
plt.legend(title="Indicator & Country", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
