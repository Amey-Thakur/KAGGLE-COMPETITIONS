import pandas as pd
import numpy as np
import os

def clean_sqft(value):
    if isinstance(value, float) or isinstance(value, int):
        return float(value)
    try:
        val_str = str(value)
        if '-' in val_str:
            parts = val_str.split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        return float(val_str)
    except:
        return np.nan

# Load the original data from the temp folder
df = pd.read_csv("temp_data/bengaluru_house_prices.csv")

# Clean the total_sqft column
df['total_sqft_cleaned'] = df['total_sqft'].apply(clean_sqft)

# Drop redundant or fully null rows in the key features
df = df.dropna(subset=['total_sqft_cleaned', 'bath', 'price'])

# Save the cleaned file
output_dir = "cleaned_dataset"
os.makedirs(output_dir, exist_ok=True)
df.to_csv(f"{output_dir}/bengaluru_house_prices_cleaned.csv", index=False)

print(f"Cleaned dataset created with {len(df)} rows.")
