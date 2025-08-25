# run_SQL_cleaning.py
import os
import sys
# Import from the newly named 'SQL_Data_Cleaners' file
from SQL_Data_Cleaners import clean_five_preventable_crime_cases_csv, clean_m891481_csv

# 1) Raw SQL datasets are stored in the "Raw_SQL" folder
RAW_DATA_DIR = "Raw_SQL"
# 2) Cleaned SQL datasets are stored in the "SQL_Dataset" folder
CLEANED_DATA_DIR = "SQL_Dataset"

def main():
    """
    Main function to orchestrate the cleaning of raw CSV files.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(CLEANED_DATA_DIR):
        os.makedirs(CLEANED_DATA_DIR)
        print(f"Created output directory: '{CLEANED_DATA_DIR}'")

    # Check if the raw data directory exists
    if not os.path.isdir(RAW_DATA_DIR):
        print(f"Error: Raw data directory '{RAW_DATA_DIR}' not found.", file=sys.stderr)
        print("Please create the 'Raw_SQL' directory and place your raw CSV files inside.", file=sys.stderr)
        return

    print(f"Starting cleaning process. Reading from '{RAW_DATA_DIR}'...")

    for filename in os.listdir(RAW_DATA_DIR):
        if not filename.endswith(".csv"):
            continue

        file_path = os.path.join(RAW_DATA_DIR, filename)
        output_file_path = os.path.join(CLEANED_DATA_DIR, f"cleaned_{filename}")

        cleaned_df = None
        try:
            if "FivePreventableCrimeCases" in filename:
                print(f"- Cleaning '{filename}' with specific function...")
                cleaned_df = clean_five_preventable_crime_cases_csv(file_path)
            elif "M891481" in filename:
                print(f"- Cleaning '{filename}' with specific function...")
                cleaned_df = clean_m891481_csv(file_path)
            else:
                print(f"- No specific cleaning function for '{filename}'. Skipping.")
                continue

            if cleaned_df is not None:
                cleaned_df.to_csv(output_file_path, index=False)
                print(f"  Successfully cleaned '{filename}'. Saved to '{output_file_path}'")
            
        except Exception as e:
            print(f"  An error occurred while cleaning '{filename}'. See error above. Skipping file.", file=sys.stderr)
            continue

if __name__ == "__main__":
    main()
    print(f"\nCleaning process finished. Cleaned files are in '{CLEANED_DATA_DIR}'.")

