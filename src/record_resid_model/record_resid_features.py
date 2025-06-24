import os
import glob
import ast
import pandas as pd
import numpy as np


def parse_time_to_seconds(time_str):
    """
    Convert a time string (e.g., "1:52.98", "59.23", "8:04.79") to total seconds (float).
    Supports "M:SS.ss", "MM:SS.ss", or "SS.ss".
    Returns np.nan if parsing fails or time_str is empty/None.
    """
    if not isinstance(time_str, str) or time_str.strip() == "":
        return np.nan

    time_str = time_str.strip()
    try:
        if ':' in time_str:
            minutes, seconds = time_str.split(':', 1)
            return float(minutes) * 60 + float(seconds)
        else:
            return float(time_str)
    except ValueError:
        return np.nan


def extract_record_time(records_str, record_type):
    """
    Given the 'records' field as a stringified Python list of lists,
    find the first occurrence of a record where record[0] == record_type
    (case-insensitive) and return its time (converted to seconds).
    If no matching record is found, return np.nan.
    """
    if not isinstance(records_str, str) or records_str.strip() == "":
        return np.nan

    try:
        records = ast.literal_eval(records_str)
    except (ValueError, SyntaxError):
        return np.nan

    target = record_type.strip().lower()
    for rec in records:
        if len(rec) >= 2 and rec[0].strip().lower() == target:
            return parse_time_to_seconds(rec[1])
    return np.nan


def load_and_split_event_files(directory_path, record_type):
    """
    Load all CSVs under directory_path matching "*.csv" that represent event-level data.
    For each row, parse 'entries' (stringified list of lists) into a separate entries DataFrame.
    Returns two DataFrames:
      - df_events: event-level info with columns [event_id, meet, stroke, gender, distance, record_time_sec, year]
      - df_entries: entry-level info with columns [event_id, athlete, seed_time_sec, final_time_sec, improvement_sec, rank]
    """
    event_csv_paths = glob.glob(os.path.join(directory_path, '*.csv'))
    df_events_list = []
    df_entries_list = []

    for path in event_csv_paths:
        try:
            df_raw = pd.read_csv(path)
            meet_id = os.path.splitext(os.path.basename(path))[0]
            
            required_cols = {'event', 'meet', 'stroke', 'gender', 'distance', 'records', 'entries'}
            if not required_cols.issubset(df_raw.columns):
                continue

            # Process each event row
            for _, row in df_raw.iterrows():
                # Create standardized event identifier from distance, gender, stroke
                stroke = str(row['stroke']).strip().title()
                gender = str(row['gender']).strip().title()
                distance = pd.to_numeric(row['distance'], errors='coerce')
                event_id = f"{distance}_{gender}_{stroke}"
                
                # Create unique meet-event combination
                unique_event_id = f"{meet_id}_{event_id}"
                
                meet = str(row['meet']).strip()
                records_str = row['records']
                entries_str = row['entries']

                # Extract specified record time
                record_time_sec = extract_record_time(records_str, record_type)

                # Extract year from meet name
                def extract_year(text):
                    for token in text.split():
                        if token.isdigit() and len(token) == 4:
                            return int(token)
                    return np.nan
                year = extract_year(meet)

                df_events_list.append({
                    'unique_event_id': unique_event_id,
                    'event_id': event_id,  # Standardized event identifier
                    'meet': meet,
                    'meet_id': meet_id,
                    'stroke': stroke,
                    'gender': gender,
                    'distance': distance,
                    'record_time_sec': record_time_sec,
                    'year': year
                })

                # Parse entries for this event
                if isinstance(entries_str, str) and entries_str.strip() != "":
                    try:
                        entries = ast.literal_eval(entries_str)
                    except (ValueError, SyntaxError):
                        entries = []
                else:
                    entries = []

                for entry in entries:
                    # Expect entry = [rank, athlete_name, age, team, seed_time_str, final_time_str]
                    if len(entry) < 6:
                        continue
                    rank = entry[0]
                    athlete = str(entry[1]).strip()
                    seed_time_str = entry[-2]
                    final_time_str = entry[-1]

                    seed_time_sec = parse_time_to_seconds(seed_time_str)
                    final_time_sec = parse_time_to_seconds(final_time_str)
                    improvement_sec = seed_time_sec - final_time_sec if (
                        not np.isnan(seed_time_sec) and not np.isnan(final_time_sec)
                    ) else np.nan

                    df_entries_list.append({
                        'unique_event_id': unique_event_id,
                        'event_id': event_id,  # Standardized event identifier
                        'meet_id': meet_id,
                        'athlete': athlete,
                        'seed_time_sec': seed_time_sec,
                        'final_time_sec': final_time_sec,
                        'improvement_sec': improvement_sec,
                        'rank': rank
                    })

        except Exception:
            continue

    if not df_events_list:
        raise FileNotFoundError(f"No valid event CSVs found in {directory_path}")

    df_events = pd.DataFrame(df_events_list)
    df_entries = pd.DataFrame(df_entries_list)

    # Cast stroke and gender to categorical
    df_events['stroke'] = df_events['stroke'].astype('category')
    df_events['gender'] = df_events['gender'].astype('category')

    print("\nDataset Summary:")
    print(f"Total events: {len(df_events)}")
    print(f"Unique event types: {df_events['event_id'].nunique()}")
    print(f"Total entries: {len(df_entries)}")
    print(f"Unique meets: {df_events['meet_id'].nunique()}")

    return df_events, df_entries


def merge_event_and_entries(df_events, df_entries):
    """Merge event and entry data using the unique event identifier."""
    
    # Remove events with no entries or missing record times
    df_events = df_events.dropna(subset=['record_time_sec'])
    
    # Group and aggregate entries, dropping NaN values
    df_entries_agg = df_entries.groupby(['unique_event_id']).agg({
        'seed_time_sec': ['count', 'mean', 'min', 'std'],
        'final_time_sec': ['mean', 'min', 'std']
    }).reset_index()
    
    # Drop events with missing entry data
    df_entries_agg = df_entries_agg.dropna()
    
    # Flatten column names
    df_entries_agg.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] 
        for col in df_entries_agg.columns
    ]
    
    # Merge and print summary
    df_merged = pd.merge(
        df_events,
        df_entries_agg,
        on='unique_event_id',
        how='inner'
    )
    
    print("\nMerged Dataset Info:")
    print(f"Total valid events: {len(df_merged)}")
    print(f"Events dropped due to missing data: {len(df_events) - len(df_merged)}")
    
    return df_merged

def engineer_model_features(df):
    """Engineer features for predicting gap to record time."""
    
    # Create features from aggregated data
    feature_df = pd.DataFrame()
    
    # Event characteristics
    feature_df['distance'] = df['distance']

    
    # Encode stroke and gender
    feature_df = pd.concat([
        feature_df,
        pd.get_dummies(df['stroke'], prefix='stroke'),
        pd.get_dummies(df['gender'], prefix='gender')
    ], axis=1)
    
    # Entry statistics (removed age-related features)
    feature_df['num_entries'] = df['seed_time_sec_count']
    feature_df['avg_seed_time'] = df['seed_time_sec_mean']
    feature_df['fastest_seed'] = df['seed_time_sec_min']
    feature_df['seed_time_spread'] = df['seed_time_sec_std']
    
    # Target: Gap between fastest final time and record
    y = df['final_time_sec_min'] - df['record_time_sec']
    
    return feature_df, y, df