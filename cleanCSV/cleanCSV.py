import pandas as pd
import numpy as np

df = pd.read_csv("Data.csv")
original_rows = len(df)
original_cols = len(df.columns)

# Remove last column
df = df.iloc[:, :-1]

# Replace blank strings or spaces with NaN
df = df.replace(r'^\s*$', pd.NA, regex=True)

# Check for columns with over 50% missing values
missing_percentage_cols = df.isnull().mean()
columns_to_drop = missing_percentage_cols[missing_percentage_cols > 0.5].index.tolist()

# Drop columns with over 50% missing values
df = df.drop(columns=columns_to_drop)

# Check for rows with over 50% missing values
missing_percentage_rows = df.isnull().mean(axis=1)
rows_to_drop = missing_percentage_rows[missing_percentage_rows > 0.5].index.tolist()
rows_before_row_cleaning = len(df)

# Drop rows with over 50% missing values
df = df.drop(index=rows_to_drop)
rows_removed_due_to_missing = rows_before_row_cleaning - len(df)

# Handle missing values for remaining columns
for column in df.columns:
    if df[column].isnull().sum() > 0:  # If column has missing values
        if pd.api.types.is_numeric_dtype(df[column]):
            # For numerical columns - using median (more robust to outliers)
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
        else:
            # For categorical columns - using most frequent
            most_frequent = df[column].mode()
            if not most_frequent.empty:
                df[column] = df[column].fillna(most_frequent[0])
            else:
                # If no mode exists, fill with a placeholder
                df[column] = df[column].fillna('Unknown')

# Remove duplicated rows
duplicated_rows = df.duplicated().sum()
df = df.drop_duplicates()

# Drop rows where ALL values are NaN (empty rows) - though this should be rare after above processing
rows_before_all_nan = len(df)
df = df.dropna(how='all')
rows_removed_all_nan = rows_before_all_nan - len(df)

# Calculate statistics
final_rows = len(df)
final_cols = len(df.columns)
total_rows_removed = original_rows - final_rows
cols_removed = original_cols - final_cols

# Save cleaned CSV
df.to_csv("cleaned_file.csv", index=False)

print(f"Original dataset: {original_rows} rows, {original_cols} columns")
print(f"Columns removed (over 50% missing): {cols_removed}")
print(f"Rows removed due to over 50% missing values: {rows_removed_due_to_missing}")
print(f"Rows removed because all values were NaN: {rows_removed_all_nan}")
print(f"Duplicated rows removed: {duplicated_rows}")
print(f"Total rows removed during cleaning: {total_rows_removed}")
print(f"Final dataset: {final_rows} rows, {final_cols} columns")
print(f"Columns dropped due to high missing values: {columns_to_drop}")
print(f"Rows dropped due to high missing values: {len(rows_to_drop)}")
