import pandas as pd
import glob
from datetime import datetime

# Path to the folder containing the season files
folder_path = "C:/Users/HP/Desktop/MyPythonJouney/Div_data/*.xlsx"

def merge_season_files(folder_path):
    all_seasons_data = []
    for file_path in glob.glob(folder_path):
        try:
            xls = pd.ExcelFile(file_path)
            print(f"Processing file: {file_path}")
            file_data = pd.concat([pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names], ignore_index=True)
            all_seasons_data.append(file_data)
        except PermissionError:
            print(f"Permission denied for file: {file_path}. Skipping this file.")
        except Exception as e:
            print(f"An error occurred while processing file: {file_path}. Error: {e}")
    combined_data = pd.concat(all_seasons_data, ignore_index=True)
    return combined_data

merged_data = merge_season_files(folder_path)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"merged_seasons_data_{timestamp}.csv"
merged_data.to_csv(output_filename, index=False)
print(f"Merged data has been saved to '{output_filename}'")
