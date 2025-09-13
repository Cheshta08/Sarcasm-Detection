import pandas as pd

# Load the Excel file using openpyxl
excel_file_path = 'dataset-1.xlsx'  # Update with your actual file path
df = pd.read_excel(excel_file_path, engine='openpyxl')

# Remove rows where the 'text' column contains 'audio not clear'
df_cleaned = df[df['text'].str.lower() != 'audio not clear']

# Remove rows where the length of the 'text' column is less than 10 characters
df_cleaned = df_cleaned[df_cleaned['text'].str.len() >= 10]

# Save the cleaned DataFrame back to an Excel file
cleaned_excel_file_path = 'cleaned.xlsx'  # Specify the path for the cleaned file
df_cleaned.to_excel(cleaned_excel_file_path, index=False)

print(f"Cleaned data saved to {cleaned_excel_file_path}")
