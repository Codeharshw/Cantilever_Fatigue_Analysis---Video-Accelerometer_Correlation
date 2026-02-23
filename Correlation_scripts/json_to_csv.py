import pandas as pd
import json
import os

def json_to_csv():
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  USER CHANGE (optional): Update these filenames if needed.         │
    # │                                                                    │
    # │  json_filename: produced by correlate_all_indents() in            │
    # │    accelerometer_video_correlation.py. Contains Pearson/Spearman  │
    # │    values for every indent × feature pair.                        │
    # │                                                                    │
    # │  csv_filename: the output matrix this script produces.            │
    # │    Rows = correlation feature pairs, Columns = Indent 1..N        │
    # └─────────────────────────────────────────────────────────────────────┘
    json_filename = 'correlation_summary_all_indents.json'
    csv_filename = 'final_summary_matrix.csv'
    
    print(f"Reading {json_filename}...")
    
    # 1. Load the JSON data
    try:
        with open(json_filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Please create '{json_filename}' and paste your JSON data into it first.")
        return

    # 2. Extract Features (Rows) and Indents (Columns)
    # We take the features from the first entry (Indent 1)
    first_key = list(data.keys())[0]
    features_list = list(data[first_key].keys())
    
    # Sort indents numerically (1, 2, ... 25) instead of string sort (1, 10, 11...)
    indent_keys = sorted(data.keys(), key=lambda x: int(x))
    
    # 3. Build the Data Table
    # Rows will be features, Columns will be Indents
    matrix_rows = []
    
    for feature in features_list:
        row_data = {'Feature': feature}
        
        for indent_id in indent_keys:
            col_name = f"Indent {indent_id}"
            
            try:
                # ┌──────────────────────────────────────────────────────────┐
                # │  USER CHANGE (optional): Change 'pearson_r' to          │
                # │  'spearman_rho' if you prefer Spearman values in the    │
                # │  output matrix instead of Pearson.                      │
                # └──────────────────────────────────────────────────────────┘
                # Extract Pearson R
                val = data[indent_id][feature]['pearson_r']
                # Round to 2 decimal places to match your notebook style
                row_data[col_name] = round(val, 2)
            except KeyError:
                row_data[col_name] = "N/A"
        
        matrix_rows.append(row_data)

    # 4. Create DataFrame and Save
    df = pd.DataFrame(matrix_rows)
    
    # Set 'Feature' as the index/first column
    df.set_index('Feature', inplace=True)
    
    print("\n--- PREVIEW OF RESULT ---")
    print(df.iloc[:, :5]) # Print first 5 columns to verify
    
    df.to_csv(csv_filename)
    print(f"\n✅ Success! Saved matrix to '{csv_filename}'")

if __name__ == "__main__":
    json_to_csv()
