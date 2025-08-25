# SQL_Data_Cleaners.py
import pandas as pd
import re
import os

def clean_five_preventable_crime_cases_csv(file_path: str) -> pd.DataFrame:
    """
    Cleans the 'FivePreventableCrimeCasesRecordedByNeighbourhoodPoliceCentreNPCAnnual.csv'
    file by transforming it into a long format with columns:
    'division', 'crime_type', 'year', and 'cases'.
    """
    # This function remains unchanged as it worked correctly.
    try:
        df = pd.read_csv(file_path, header=None)
        df.rename(columns={0: 'DataSeries'}, inplace=True)
        year_row = df.iloc[0, 1:]
        year_cols_raw = [col for col in year_row if isinstance(col, (int, float)) or (isinstance(col, str) and col.isdigit())]
        years = [str(int(float(str(y)))) for y in year_cols_raw]
        new_columns = ['DataSeries'] + years
        df.columns = new_columns
        df = df.iloc[1:].copy()
        df.replace(['na', ''], pd.NA, inplace=True)

        cleaned_df_list = []
        current_division = None
        for index, row in df.iterrows():
            data_series_val = str(row['DataSeries']).strip()
            if 'Police Division - Total' in data_series_val or 'Police Division - ' in data_series_val:
                current_division = data_series_val
            elif current_division and pd.notna(data_series_val) and data_series_val != 'nan':
                crime_type = data_series_val
                for year in years:
                    cases = row[year]
                    cleaned_df_list.append({
                        'division': current_division,
                        'crime_type': crime_type,
                        'year': int(year),
                        'cases': cases
                    })
        cleaned_df = pd.DataFrame(cleaned_df_list)
        cleaned_df['cases'] = pd.to_numeric(cleaned_df['cases'], errors='coerce').fillna(0).astype(int)
        cleaned_df['division'] = cleaned_df['division'].str.replace(' - Total', '', regex=False).str.lower().str.strip()
        cleaned_df['crime_type'] = cleaned_df['crime_type'].str.lower().str.strip()
        return cleaned_df
    except Exception as e:
        print(f"Error cleaning {os.path.basename(file_path)}: {e}")
        raise

def clean_m891481_csv(file_path: str) -> pd.DataFrame:
    """
    Cleans the 'M891481.csv' file by skipping metadata, identifying the correct header,
    and transforming it into a long format with columns:
    'metric', 'unit', 'year', and 'value'.
    """
    try:
        df = pd.read_csv(file_path, header=None)
        header = df.iloc[9].tolist()
        data = df.iloc[10:].copy()
        data.columns = header
        data = data.dropna(axis=1, how='all')
        new_cols = ['data_series' if isinstance(col, str) and col.strip() == 'Data Series' else col for col in data.columns]
        data.columns = new_cols

        # --- THIS IS THE CORRECTED LINE ---
        # By adding str(col), we ensure that any non-string values (like NaN) are converted
        # to their string representation before .strip() is called, preventing the error.
        data.columns = [re.sub(r'[^a-z0-9_]+', '', str(col).strip().lower().replace(' ', '_')) for col in data.columns]
        
        id_vars = ['data_series']
        value_vars = [col for col in data.columns if re.fullmatch(r'\d{4}', col)]
        cleaned_df = data.melt(id_vars=id_vars, value_vars=value_vars, var_name='year', value_name='value')
        cleaned_df['year'] = pd.to_numeric(cleaned_df['year'], errors='coerce').astype(int)
        cleaned_df['metric'] = cleaned_df['data_series'].apply(lambda x: x.split('(')[0].strip().lower() if isinstance(x, str) else x)
        cleaned_df['unit'] = cleaned_df['data_series'].apply(lambda x: x.split('(')[1].replace(')', '').strip().lower() if isinstance(x, str) and '(' in x else 'number_of_cases_recorded')
        cleaned_df.drop(columns=['data_series'], inplace=True)
        cleaned_df['value'] = cleaned_df['value'].astype(str).str.replace(',', '', regex=False)
        cleaned_df['value'] = pd.to_numeric(cleaned_df['value'], errors='coerce').fillna(0).astype(int)
        return cleaned_df
    except Exception as e:
        print(f"Error cleaning {os.path.basename(file_path)}: {e}")
        raise
