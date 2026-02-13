
import pandas as pd
import os

file_path = "EXCEL_CONSOLIDATO_FINALE.xlsx"

try:
    xl = pd.ExcelFile(file_path)
    print("Sheet names:", xl.sheet_names)
    
    # Iterate specifically looking for ablation or rank info
    found = False
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        print(f"\nScanning Sheet: {sheet}")
        print(df.head())
        
        # Check for rank related columns or values
        # Convert to string to concise search
        df_str = df.to_string()
        if "rank" in df_str.lower() or "r=" in df_str.lower() or "3.85" in df_str:
            print(f"!!! FOUND POTENTIAL MATCH IN SHEET: {sheet} !!!")
            # Print rows that might contain the data
            if "3.85" in df_str:
                print("Found value 3.85 (Rank 4 Composite)")
            
            # Print the whole dataframe if small, or filtered
            print(df)
            found = True

except Exception as e:
    print(f"Error reading excel: {e}")
