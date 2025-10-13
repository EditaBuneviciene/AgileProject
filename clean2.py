import pandas as pd

# Load CSV
df = pd.read_csv("data_Copy.csv")

# Store original shape for reporting
original_rows, original_cols = df.shape

# Replace blank strings or spaces with NaN
df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

# Drop rows that are completely empty
#df.dropna(how="all", inplace=True)

# Drop columns that are completely empty
df.dropna(axis="columns", how="all", inplace=True)

# Get cleaned shape
cleaned_rows, cleaned_cols = df.shape

# Save cleaned CSV
df.to_csv("cleaned_file2.csv", index=False)

# Print summary report
print("✅ Cleaning Summary:")
print(f"  • Original file: {original_rows:,} rows × {original_cols:,} columns")
print(f"  • Cleaned file:  {cleaned_rows:,} rows × {cleaned_cols:,} columns")
print(f"  • Rows removed:  {original_rows - cleaned_rows:,}")
print(f"  • Columns removed: {original_cols - cleaned_cols:,}")
print("\nCleaned file saved as: cleaned_file2.csv")