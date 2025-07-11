import pandas as pd
import ast
import json
from pathlib import Path
import os
import re

# Define paths
individual_dir = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'parsed' / 'individual'
txts_dir = Path(__file__).parent / 'txts_to_update'

def parse_txt_to_csv(txt_filepath):
    """Parse a txt file back into CSV format"""
    events = []
    with open(txt_filepath, 'r') as f:
        content = f.read()
    
    # Split by the separator line
    event_sections = content.split('-' * 50)
    
    for section_idx, section in enumerate(event_sections):
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        if not lines:
            continue
            
        event_data = {}
        current_line = 0
        
        # Parse event info (lines that don't start with "Swimmer" or "Results")
        while current_line < len(lines):
            line = lines[current_line].strip()
            if line.startswith('Event '):
                pass  # Skip event number line
            elif line.startswith('Swimmer '):
                # We've reached the swimmer data section
                break
            elif ': ' in line:
                # This is an event property
                key, value = line.split(': ', 1)
                event_data[key] = value
            current_line += 1
        
        # Parse swimmer data
        swimmers = []
        current_swimmer = {}
        
        while current_line < len(lines):
            line = lines[current_line]
            
            if line.startswith('Swimmer '):
                # Save previous swimmer if exists
                if current_swimmer:
                    swimmers.append(current_swimmer)
                current_swimmer = {}
            elif ': ' in line and not line.startswith('Swimmer '):
                # This is a swimmer property
                key_value = line.split(': ', 1)
                if len(key_value) == 2:
                    key, value = key_value
                    # Convert value types
                    if value == 'None':
                        value = None
                    elif value == 'True':
                        value = True
                    elif value == 'False':
                        value = False
                    elif key in ['prelim_rank', 'final_rank'] and value.isdigit():
                        value = int(value)
                    elif key in ['prelim_time', 'finals_time', 'seed_time']:
                        # Keep time values as strings, but clean them up
                        if value != 'None':
                            # Remove any extra whitespace or formatting
                            value = value.strip()
                    elif value.isdigit():
                        value = int(value)
                    current_swimmer[key] = value
            
            current_line += 1
        
        # Don't forget the last swimmer
        if current_swimmer:
            swimmers.append(current_swimmer)
        
        event_data['results'] = swimmers
        events.append(event_data)
    
    return events

def main():
    if not txts_dir.exists():
        print("txts_to_update directory does not exist!")
        return
    
    txt_files = list(txts_dir.glob('*_event_results.txt'))
    if not txt_files:
        print("No txt files found in txts_to_update directory!")
        return
    
    print(f"Found {len(txt_files)} txt files to process")
    new_csv_files = []
    
    for txt_file in txt_files:
        print(f"Processing {txt_file.name}...")
        try:
            events = parse_txt_to_csv(txt_file)
            df = pd.DataFrame(events)
            
            # Generate filename
            match = re.match(r'(\d{4})-NESCAC-MSD-Results', txt_file.stem)
            if match:
                year = match.group(1)
                base_name = f"{year}-NESCAC-MSD-Results_parsed_updated"
            else:
                base_name = txt_file.stem.replace('_event_results', '_parsed_updated')
            
            csv_filename = f"{base_name}.csv"
            csv_filepath = individual_dir / csv_filename
            df.to_csv(csv_filepath, index=False)
            print(f"Created {csv_filename}")
            new_csv_files.append(csv_filepath)
            txt_file.unlink()
            print(f"Deleted {txt_file.name}")
        except Exception as e:
            print(f"Error processing {txt_file.name}: {str(e)}")
    
    # After all new CSVs are written, delete old *_parsed.csv files (but not *_parsed_updated.csv)
    old_csvs = [f for f in individual_dir.glob('*_parsed.csv') if not str(f).endswith('_parsed_updated.csv')]
    for old_csv in old_csvs:
        try:
            old_csv.unlink()
            print(f"Deleted old CSV: {old_csv.name}")
        except Exception as e:
            print(f"Error deleting {old_csv.name}: {str(e)}")
    
    # Rename *_parsed_updated.csv to *_parsed.csv
    updated_csvs = list(individual_dir.glob('*_parsed_updated.csv'))
    for updated_csv in updated_csvs:
        new_name = str(updated_csv).replace('_parsed_updated.csv', '_parsed.csv')
        Path(updated_csv).rename(new_name)
        print(f"Renamed {updated_csv.name} to {Path(new_name).name}")
    
    print("Processing complete! All txt files have been converted back to CSV, old CSVs removed, and new files renamed.")

if __name__ == "__main__":
    main() 