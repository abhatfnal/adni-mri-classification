import pandas as pd

# Define the input Excel file from your 'ls' command
excel_file = '/project/aereditato/abhat/OASIS/OASIS_1/oasis_cross-sectional-5708aa0a98d82080.xlsx'

# Define the clean output CSV file name
csv_file = 'oasis_cross_sectional.csv'

try:
    # Read the Excel file. You might need to have 'openpyxl' installed.
    # If not, run: pip install pandas openpyxl
    df = pd.read_excel(excel_file)

    # Save the data to a CSV file, without the pandas index column
    df.to_csv(csv_file, index=False)

    print(f"✅ Successfully converted '{excel_file}' to '{csv_file}'")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please make sure you have pandas and openpyxl installed in your environment.")