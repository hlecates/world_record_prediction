import pandas as pd
import os
from pathlib import Path
import re

# Define paths
individual_dir = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'parsed' / 'individual'
txts_dir = Path(__file__).parent / 'txts_to_update'

def write_txt_for_csv(csv_filepath, txt_filepath):
    df = pd.read_csv(csv_filepath)
    with open(txt_filepath, 'w') as f:
        for idx, row in df.iterrows():
            for col in df.columns:
                if col != 'results':
                    f.write(f"{col}: {row[col]}\n")
            # Write results (swimmers)
            results_value = row.get('results')
            if results_value is not None and pd.notna(results_value):
                try:
                    results_str = str(results_value)
                    swimmers = eval(results_str) if isinstance(results_value, str) else results_value
                except Exception:
                    swimmers = []
                if isinstance(swimmers, list) and swimmers:
                    f.write("Results:\n")
                    for i, swimmer in enumerate(swimmers, 1):
                        f.write(f"\nSwimmer {i}:\n")
                        for key, value in swimmer.items():
                            f.write(f"{key}: {value}\n")
                else:
                    f.write("Results:\n")
            else:
                f.write("Results:\n")
            f.write("--------------------------------------------------\n")

def main():
    if not txts_dir.exists():
        txts_dir.mkdir(parents=True)
    csv_files = list(individual_dir.glob('*_parsed.csv'))
    if not csv_files:
        print("No CSV files found in individual directory!")
        return
    print(f"Found {len(csv_files)} CSV files to process")
    for csv_file in csv_files:
        base_name = csv_file.stem.replace('_parsed', '_event_results')
        txt_file = txts_dir / f"{base_name}.txt"
        write_txt_for_csv(csv_file, txt_file)
        print(f"Created {txt_file.name}")
    print("Processing complete! All CSV files have been converted to txt files in the txts_to_update directory.")

if __name__ == "__main__":
    main()