import pandas as pd
df = pd.read_csv("data_Copy.csv")

# Replace blank strings or spaces with NaN
df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

# Drop rows where ALL values are NaN (empty row)
df.dropna(how="any", inplace=True)
df.dropna(axis="columns")
# Save cleaned CSV
df.to_csv("cleaned_file.csv", index=False)