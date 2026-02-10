import pandas as pd

# Read the demographics Excel file
file_path = r"c:\Users\ashut\Downloads\4th Year\Speech_Healthy_PD[1]\Speech_Healthy_PD[1]\23849127\Demographics_age_sex.xlsx"
df = pd.read_excel(file_path)

# Display basic information about the dataset
print("Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

print("\nBasic statistics:")
print(df.describe())

print("\nValue counts for categorical variables:")
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"\n{col}:")
        print(df[col].value_counts())