import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def extract_time_seconds(time_str: str) -> float:
    """Convert time string (MM:SS.ss or SS.ss) to seconds."""
    if ':' in time_str:
        minutes, seconds = time_str.split(':')
        return float(minutes) * 60 + float(seconds)
    return float(time_str)

def load_meet_results() -> pd.DataFrame:
    """Load and combine all parsed meet results."""
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / "data" / "processed" / "parsed_meet_results" 
    
    dfs = []
    for csv_file in processed_dir.glob("*_results.csv"):
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def validate_event_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out events with invalid or missing data."""
    
    initial_count = len(df)
    print(f"\nInitial event count: {initial_count}")
    
    # Convert lists from strings if needed
    df['records'] = df['records'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df['entries'] = df['entries'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    # Check each condition separately and report
    world_records = df['records'].apply(lambda x: any(r[0] == 'World' for r in x))
    american_records = df['records'].apply(lambda x: any(r[0] == 'American' for r in x))
    us_open_records = df['records'].apply(lambda x: any(r[0] == 'U.S. Open' for r in x))
    min_entries = df['entries'].apply(len) >= 2  # Reduced from 3 to 2
    valid_times = df['entries'].apply(
        lambda x: any(e[4] is not None and e[5] is not None for e in x)
    )  # Changed from all to any
    basic_data = df['distance'].notna() & df['stroke'].notna() & df['gender'].notna()
    
    # Print detailed filtering stats
    print("\nFiltering Statistics:")
    print(f"Missing World Records: {(~world_records).sum()}")
    print(f"Missing American Records: {(~american_records).sum()}")
    print(f"Missing US Open Records: {(~us_open_records).sum()}")
    print(f"Events with < 2 entries: {(~min_entries).sum()}")
    print(f"Events with no valid times: {(~valid_times).sum()}")
    print(f"Events missing basic data: {(~basic_data).sum()}")
    
    # Combined filter with less strict criteria
    valid_events = (
        (world_records | american_records | us_open_records) &  # Any record type
        min_entries &
        valid_times &
        basic_data
    )
    
    df_valid = df[valid_events].copy()

    print(f"\nEvents removed: {initial_count - len(df_valid)}")
    print(f"Final event count: {len(df_valid)}")
    
    return df_valid

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from meet results data with focus on record prediction."""
    
    # First validate and filter the data
    df = validate_event_data(df)
    
    # Records and entries are already converted to lists by validate_event_data
    # Remove the duplicate conversion
    
    # Extract record times and convert to seconds
    df['world_record'] = df['records'].apply(
        lambda x: min([extract_time_seconds(r[1]) for r in x if r[0] == 'World'], default=np.nan)
    )
    df['american_record'] = df['records'].apply(
        lambda x: min([extract_time_seconds(r[1]) for r in x if r[0] == 'American'], default=np.nan)
    )
    df['us_open_record'] = df['records'].apply(
        lambda x: min([extract_time_seconds(r[1]) for r in x if r[0] == 'U.S. Open'], default=np.nan)
    )
    
    # Create target variables (1 if record broken, 0 if not)
    df['world_record_broken'] = df.apply(
        lambda row: 1 if any(extract_time_seconds(e[5]) < row['world_record'] 
                            for e in row['entries']) else 0, axis=1
    )
    df['american_record_broken'] = df.apply(
        lambda row: 1 if any(extract_time_seconds(e[5]) < row['american_record'] 
                            for e in row['entries']) else 0, axis=1
    )
    df['us_open_record_broken'] = df.apply(
        lambda row: 1 if any(extract_time_seconds(e[5]) < row['us_open_record'] 
                            for e in row['entries']) else 0, axis=1
    )
    
    # Features from seed times only
    df['num_entries'] = df['entries'].apply(len)
    df['fastest_seed'] = df['entries'].apply(
        lambda x: min([extract_time_seconds(e[4]) for e in x if e[4] is not None])
    )
    df['avg_seed_time'] = df['entries'].apply(
        lambda x: np.mean([extract_time_seconds(e[4]) for e in x if e[4] is not None])
    )
    df['seed_time_std'] = df['entries'].apply(
        lambda x: np.std([extract_time_seconds(e[4]) for e in x if e[4] is not None])
    )
    
    # Percentages off records (using seed times)
    df['fastest_seed_pct_off_world'] = ((df['fastest_seed'] - df['world_record']) 
                                       / df['world_record'] * 100)
    df['fastest_seed_pct_off_american'] = ((df['fastest_seed'] - df['american_record']) 
                                          / df['american_record'] * 100)
    
    # Event characteristics
    df['is_sprint'] = df['distance'] <= 100
    df['is_distance'] = df['distance'] >= 800
    df['stroke_encoded'] = pd.Categorical(df['stroke']).codes
    
    # Add record holder features
    def is_record_holder(entries, records):
        record_holders = set(r[3] for r in records)  # Get all record holder names
        # Sort entries by seed time using extract_time_seconds
        sorted_entries = sorted(
            [e for e in entries if e[4]], 
            key=lambda x: extract_time_seconds(x[4])
        )
        if not sorted_entries:  # Handle case with no valid entries
            return 0
        top_seed_name = sorted_entries[0][1]  # Get name of fastest seed
        return 1 if top_seed_name in record_holders else 0
    
    df['top_seed_is_record_holder'] = df.apply(
        lambda row: is_record_holder(row['entries'], row['records']), axis=1
    )

    def get_position_stats(entries, record_time, positions=[1, 3, 5, 8]):
        # Sort entries by seed time
        sorted_entries = sorted(
            [e for e in entries if e[4]], 
            key=lambda x: extract_time_seconds(x[4])
        )
        
        stats = {}
        for pos in positions:
            if pos <= len(sorted_entries):
                seed_time = extract_time_seconds(sorted_entries[pos-1][4])
                stats[f'pos_{pos}_time_gap'] = (seed_time - record_time) / record_time * 100
            else:
                stats[f'pos_{pos}_time_gap'] = np.nan
        return stats
    
    # Calculate time gaps for various positions relative to records
    for record_type in ['world', 'american', 'us_open']:
        record_times = df[f'{record_type}_record']
        position_stats = df.apply(
            lambda row: get_position_stats(row['entries'], row[f'{record_type}_record']), 
            axis=1
        )
        
        for pos in [1, 3, 5, 8]:
            df[f'pos_{pos}_pct_off_{record_type}'] = position_stats.apply(
                lambda x: x.get(f'pos_{pos}_time_gap')
            )
    
    # Add field strength indicators
    df['top_3_spread'] = df.apply(
        lambda row: np.ptp([
            extract_time_seconds(e[4]) for e in sorted(
                [e for e in row['entries'] if e[4]], 
                key=lambda x: extract_time_seconds(x[4])
            )[:3]
        ]) if len(row['entries']) >= 3 else np.nan, 
        axis=1
    )

    # Add time of day/year features
    df['season_peak'] = df['meet'].str.contains('Olympic|World|National', case=False).astype(int)
    
    # Competitive field features
    df['num_sub_1pct_off_record'] = df.apply(
        lambda row: sum(
            1 for e in row['entries'] 
            if e[4] and (extract_time_seconds(e[4]) - row['world_record']) / row['world_record'] * 100 < 1
        ), 
        axis=1
    )

    # Drop original records and entries lists
    df = df.drop(['records', 'entries'], axis=1)
    
    return df

def main():
    print("Loading meet results...")
    df = load_meet_results()
    
    print("\nEngineering features...")
    df_features = engineer_features(df)
    
    # Save engineered features
    output_dir = Path(__file__).parent.parent / "data" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    output_path = output_dir / "record_prediction_features.csv"
    df_features.to_csv(output_path, index=False)
    
    # Enhanced summary statistics
    print("\nDetailed Feature Summary:")
    print(f"Total events processed: {len(df_features)}")
    
    # Record breaking statistics
    print("\nRecord Breaking Stats:")
    print(f"World records broken: {df_features['world_record_broken'].sum()} ({df_features['world_record_broken'].mean()*100:.1f}%)")
    print(f"American records broken: {df_features['american_record_broken'].sum()} ({df_features['american_record_broken'].mean()*100:.1f}%)")
    print(f"US Open records broken: {df_features['us_open_record_broken'].sum()} ({df_features['us_open_record_broken'].mean()*100:.1f}%)")
    
    # Event type distribution
    print("\nEvent Distribution:")
    print(f"Sprint events: {df_features['is_sprint'].sum()} ({df_features['is_sprint'].mean()*100:.1f}%)")
    print(f"Distance events: {df_features['is_distance'].sum()} ({df_features['is_distance'].mean()*100:.1f}%)")
    print(f"Middle distance events: {len(df_features) - df_features['is_sprint'].sum() - df_features['is_distance'].sum()}")
    
    # Time analysis
    print("\nTime Analysis:")
    print(f"Average % off world record (fastest seed): {df_features['fastest_seed_pct_off_world'].mean():.2f}%")
    print(f"Average % off american record (fastest seed): {df_features['fastest_seed_pct_off_american'].mean():.2f}%")
    print(f"Average entries per event: {df_features['num_entries'].mean():.1f}")
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()