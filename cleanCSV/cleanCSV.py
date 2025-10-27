import pandas as pd

df = pd.read_csv("Data.csv")
original_rows = len(df)
df = df.iloc[:, :-1] #remove last column
# Replace blank strings or spaces with NaN
df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

# Drop rows where ALL values are NaN (empty row)
df.dropna(how="any", inplace=True)
df.dropna(axis="columns")
cleaned_rows = original_rows - len(df)
# Save cleaned CSV
df.to_csv("cleaned_file.csv", index=False)

print(f" Original rows: {original_rows}")
print(f" Rows removed during cleaning: {cleaned_rows}")
print(f" Final rows: {len(df)}")