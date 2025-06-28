import pandas as pd
import numpy as np
from pathlib import Path
import ast
from typing import List, Dict, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class TimeConverter:
    @staticmethod
    def time_to_seconds(time_str: str) -> float:
        if pd.isna(time_str):
            return np.nan
        
        time_str = time_str.strip()

        if ':' in time_str:
            parts = time_str.split(':')
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
             return float(time_str)
        

    @staticmethod
    def seconds_to_time(seconds: float) -> str:
        if pd.isna(seconds):
            return ''
        
        if seconds >= 60:
            minutes = int(seconds // 60)
            seconds = seconds % 60
            return f"{minutes}:{seconds:05.2f}"
        else:
            return f"{seconds:05.2f}"
        

class RecordAnalyzer:
    def __init__(self):
        self.time_converter = TimeConverter()

    
    def parse_records(self, records_str: str) -> List[Dict]:
        if pd.isna(records_str) or records_str == '[]':
            return []
        
        try:
            records_list = ast.literal_eval(records_str)
            parsed_records = []
            for record in records_list:
                parsed_records.append({
                    'type': record[0],
                    'time': record[1],
                    'date': record[2],
                    'athlete': record[3],
                    'team_country': record[4]
                })
            return parsed_records
        except:
            return []
        

    def get_best_time_by_type(self, records: List[Dict], record_type: str) -> Optional[float]:
        matching_records = [r for r in records if r['type'] == record_type]

        if not matching_records:
            return None
        
        times = []
        for record in matching_records:
            time = self.time_converter.time_to_seconds(record['time'])
            if not pd.isna(time):
                times.append(time) 

        return min(times) if times else None
    

class SeedTimeAnalyzer:
    def __init__(self):
        self.time_converter = TimeConverter()

    
    def extract_seed_time(self, entries: List[Dict]) -> Optional[float]:
        seed_times = []

        for entry in entries:
            seed_time = self.time_converter.time_to_seconds(entry['seed_time'])
            if not pd.isna(seed_time):
                seed_times.append(seed_time)

        return sorted(seed_times)
    

    def calculate_field_depth_features(self, seed_times: List[float]) -> Dict[str, float]:
        features = {}

        if len(seed_times) == 0:
            return self._empty_field_features()
        
        # Basic statistics
        features['field_size'] = len(seed_times)
        features['seed_mean'] = np.mean(seed_times)
        features['seed_median'] = np.median(seed_times)
        features['seed_std'] = np.std(seed_times)
        features['seed_cv'] = features['seed_std'] / features['seed_mean'] if features['seed_mean'] > 0 else 0

        # Distribution features
        if len(seed_times) >= 3:
            features['seed_skewness'] = stats.skew(seed_times)
            features['seed_kurtosis'] = stats.kurtosis(seed_times)
        else:
            features['seed_skewness'] = 0
            features['seed_kurtosis'] = 0

        # Percentile features
        if len(seed_times) >= 4:
            q25, q75 = np.percentile(seed_times, [25, 75])
            features['seed_iqr'] = q75 - q25
            features['seed_iqr_ratio'] = features['seed_iqr'] / features['seed_median']
        else:
            features['seed_iqr'] = 0
            features['seed_iqr_ratio'] = 0

        # Ratio features
        if len(seed_times) >= 5:
            features['seed_5th_to_1st_ratio'] = seed_times[4] / seed_times[0]
        else:
            features['seed_5th_to_1st_ratio'] = np.nan
        
        if len(seed_times) >= 3:
            features['seed_3rd_to_1st_ratio'] = seed_times[2] / seed_times[0]
        else:
            features['seed_3rd_to_1st_ratio'] = np.nan

        # Gap analysis
        if len(seed_times) >= 2:
            gaps = [seed_times[i+1] - seed_times[i] for i in range(len(seed_times)-1)]
            features['max_gap'] = max(gaps)
            features['avg_gap'] = np.mean(gaps)
            features['gap_1st_2nd'] = gaps[0]
            
            # Find biggest gap position
            max_gap_idx = gaps.index(max(gaps))
            features['max_gap_position'] = max_gap_idx + 1
        else:
            features['max_gap'] = 0
            features['avg_gap'] = 0
            features['gap_1st_2nd'] = 0
            features['max_gap_position'] = 0

        # Herfindahl-Hirschman Index (HHI) for seed times
        # Convert times to shares by inverting and normalizing
        if len(seed_times) > 1:
            inverted_times = [1/t for t in seed_times]
            total = sum(inverted_times)
            shares = [t/total for t in inverted_times]
            features['hhi_seed_times'] = sum(s**2 for s in shares)
        else:
            features['hhi_seed_times'] = 1.0
        
        return features
    

    def calculate_record_proximity_features(self, seed_times: List[float], 
                                            world_record: Optional[float], 
                                            american_record: Optional[float], 
                                            us_open_record: Optional[float]) -> Dict:
        features = {}

        if len(seed_times) == 0:
            return self._empty_record_proximity_features()
        
        fastest_seed = seed_times[0]

        records = {
            'world': world_record,
            'american': american_record,
            'us_open': us_open_record
        }

        for record_name, record_time in records.items():
            if record_time:
                gap = record_time - fastest_seed
                features[f'top_seed_vs_{record_name}_record'] = gap
                features[f'top_seed_{record_name}_record_pct'] = gap / record_time

                # Count swimmers within certain thresholds
                for threshold in [0.02, 0.03, 0.05]:
                    within_thresh_distance = record_time * (1 + threshold)
                    count_within_thresh = sum(1 for t in seed_times if t <= within_thresh_distance)
                    features[f'swimmers_within_{int(threshold*100)}pct_{record_name}'] = count_within_thresh
            else:
                features[f'top_seed_vs_{record_name}_record'] = np.nan
                features[f'top_seed_{record_name}_record_pct'] = np.nan
                for threshold in [0.02, 0.03, 0.05]:
                    features[f'swimmers_within_{int(threshold*100)}pct_{record_name}'] = 0

        # Look at clustering around the top seed
        for threshold in [0.05, 0.10, 0.15]:
            top_seed_threshold = fastest_seed * (1 + threshold)
            count_within_thresh = sum(1 for t in seed_times if t <= top_seed_threshold)
            features[f'swimmers_within_{int(threshold*100)}pct_top_seed'] = count_within_thresh

        return features
    

    def calculate_psychological_features(self, seed_times: List[float]) -> Dict:
        features = {}
        
        if len(seed_times) < 2:
            return self._empty_psychological_features()
        
        # Pressure index ie how dominant is the top seed?
        gap_to_second = seed_times[1] - seed_times[0]
        features['pressure_index'] = gap_to_second / seed_times[0]
        
        # Dark horse potential ie strength of 3rd-8th place relative to top 2
        if len(seed_times) >= 8:
            top_2_avg = np.mean(seed_times[:2])
            mid_field_avg = np.mean(seed_times[2:8])
            features['dark_horse_potential'] = (mid_field_avg - top_2_avg) / top_2_avg
        else:
            features['dark_horse_potential'] = np.nan
        
        # Time span of middle 50%
        if len(seed_times) >= 4:
            q25_idx = len(seed_times) // 4
            q75_idx = 3 * len(seed_times) // 4
            competitive_bandwidth = seed_times[q75_idx] - seed_times[q25_idx]
            features['competitive_bandwidth'] = competitive_bandwidth
            features['competitive_bandwidth_pct'] = competitive_bandwidth / seed_times[0]
        else:
            features['competitive_bandwidth'] = np.nan
            features['competitive_bandwidth_pct'] = np.nan
        
        return features


    def _empty_field_features(self) -> Dict[str, float]:
        return {
            'field_size': 0,
            'seed_mean': np.nan,
            'seed_median': np.nan,
            'seed_std': np.nan,
            'seed_cv': np.nan,
            'seed_skewness': np.nan,
            'seed_kurtosis': np.nan,
            'seed_iqr': np.nan,
            'seed_iqr_ratio': np.nan,
            'seed_5th_to_1st_ratio': np.nan,
            'seed_3rd_to_1st_ratio': np.nan,
            'max_gap': np.nan,
            'avg_gap': np.nan,
            'gap_1st_2nd': np.nan,
            'max_gap_position': np.nan,
            'hhi_seed_times': np.nan
        }
    

    def _empty_record_proximity_features(self) -> Dict:
        features = {}
        for record_name in ['world', 'american', 'us_open']:
            features[f'top_seed_vs_{record_name}_record'] = np.nan
            features[f'top_seed_{record_name}_record_pct'] = np.nan
            for threshold in [2, 3, 5]:
                features[f'swimmers_within_{threshold}pct_{record_name}'] = 0
        
        for threshold in [5, 10, 15]:
            features[f'swimmers_within_{threshold}pct_top_seed'] = 0
        
        return features
    

    def _empty_psychological_features(self) -> Dict:
        return {
            'pressure_index': np.nan,
            'dark_horse_potential': np.nan,
            'competitive_bandwidth': np.nan,
            'competitive_bandwidth_pct': np.nan
        }
    

class ParticipantAnalyzer:

    def analyze_participants(self, entries: List[Dict], records: List[Dict]) -> Dict:
        features = {}
        
        if not entries:
            return self._empty_participant_features()
        
        # Age analysis
        ages = [entry['age'] for entry in entries if pd.notna(entry['age'])]
        if ages:
            features['avg_age'] = np.mean(ages)
            features['top_seed_age'] = entries[0]['age'] if entries else np.nan
            features['age_range'] = max(ages) - min(ages) if len(ages) > 1 else 0
        else:
            features['avg_age'] = np.nan
            features['top_seed_age'] = np.nan
            features['age_range'] = np.nan

        # Get record holders by type
        record_holders_by_type = {}
        for record in records:
            record_type = record['type']
            if record_type not in record_holders_by_type:
                record_holders_by_type[record_type] = []
            record_holders_by_type[record_type].append(record['athlete'])
        
        participant_names = [entry['name'] for entry in entries]

        # Check if specific record holders are competing
        features['world_record_holder_competing'] = any(
            name in record_holders_by_type.get('World', []) for name in participant_names
        )
        features['american_record_holder_competing'] = any(
            name in record_holders_by_type.get('American', []) for name in participant_names
        )
        features['us_open_record_holder_competing'] = any(
            name in record_holders_by_type.get('U.S. Open', []) for name in participant_names
        )

        # General record holder participation (any type)
        all_record_holders = set()
        for record in records:
            all_record_holders.add(record['athlete'])
        
        features['record_holders_count'] = sum(1 for name in participant_names if name in all_record_holders)
        features['top_seed_holds_record'] = participant_names[0] in all_record_holders if participant_names else False
    
        return features

    def _empty_participant_features(self) -> Dict:
        return {
            'avg_age': np.nan,
            'top_seed_age': np.nan,
            'age_range': np.nan,
            'world_record_holder_competing': False,
            'american_record_holder_competing': False,
            'us_open_record_holder_competing': False,
            'record_holders_count': 0,
            'top_seed_holds_record': False,
        }
    

class FeatureEngineer:
    def __init__(self):
        self.time_converter = TimeConverter()
        self.record_analyzer = RecordAnalyzer()
        self.seed_time_analyzer = SeedTimeAnalyzer()
        self.participant_analyzer = ParticipantAnalyzer()


    def parse_entries(self, entries_str: str) -> List[Dict]:
        if pd.isna(entries_str) or entries_str == '[]':
            return []
        
        try:
            entries_list = ast.literal_eval(entries_str)
            parsed_entries = []
            for entry in entries_list:
                parsed_entries.append({
                    'rank': entry[0],
                    'name': entry[1],
                    'age': entry[2],
                    'team': entry[3],
                    'seed_time': entry[4],
                    'final_time': entry[5]
                })
            return parsed_entries
        except:
            return []


    def create_event_features(self, row: pd.Series) -> Dict:
        features = {}

        # Add basic event information
        # features['event_id'] = row['event_id']
        features['meet'] = row['meet'].strip()
        features['stroke'] = row['stroke']
        features['gender'] = row['gender']
        features['distance'] = row['distance']
        features['event_type'] = f"{row['gender']}_{row['stroke']}_{row['distance']}"

        # Parse data
        records = self.record_analyzer.parse_records(row['records'])
        entries = self.parse_entries(row['entries'])

        # Get world, American, and US Open records
        world_record = self.record_analyzer.get_best_time_by_type(records, 'World')
        american_record = self.record_analyzer.get_best_time_by_type(records, 'American')
        us_open_record = self.record_analyzer.get_best_time_by_type(records, 'U.S. Open')

        features['world_record_time'] = world_record
        features['american_record_time'] = american_record
        features['us_open_record_time'] = us_open_record

        # Get seed times
        seed_times = self.seed_time_analyzer.extract_seed_time(entries)


        # Add all features
        features.update(self.seed_time_analyzer.calculate_field_depth_features(seed_times))
        features.update(self.seed_time_analyzer.calculate_record_proximity_features(
            seed_times, world_record, american_record, us_open_record))
        features.update(self.seed_time_analyzer.calculate_psychological_features(seed_times))
        features.update(self.participant_analyzer.analyze_participants(entries, records))

        # Add target features
        features.update(self._create_target_features(entries, world_record, american_record, us_open_record))

        return features
    
    
    def _create_target_features(self, entries: List[Dict],
                                world_record: Optional[float],
                                american_record: Optional[float],
                                us_open_record: Optional[float]) -> Dict:
        targets = {}
        
        # Find actual race results
        results = []
        for entry in entries:
            final_time = self.time_converter.time_to_seconds(entry['final_time'])
            seed_time = self.time_converter.time_to_seconds(entry['seed_time'])
            if not pd.isna(final_time):
                results.append({
                    'name': entry['name'],
                    'final_time': final_time,
                    'seed_time': seed_time
                })
        
        if not results:
            # No race results available
            targets['winner_vs_world_record'] = np.nan
            targets['winner_vs_american_record'] = np.nan
            targets['winner_vs_us_open_record'] = np.nan
            targets['top_seed_won'] = np.nan
            return targets
        
        # Sort by final time to find winner
        results.sort(key=lambda x: x['final_time'])
        winner = results[0]
        
        # Calculate record residuals
        if world_record:
            targets['winner_vs_world_record'] = winner['final_time'] - world_record
        else:
            targets['winner_vs_world_record'] = np.nan
            
        if american_record:
            targets['winner_vs_american_record'] = winner['final_time'] - american_record
        else:
            targets['winner_vs_american_record'] = np.nan
            
        if us_open_record:
            targets['winner_vs_us_open_record'] = winner['final_time'] - us_open_record
        else:
            targets['winner_vs_us_open_record'] = np.nan
        
        # Determine if top seed won
        seed_results = [(r['name'], r['seed_time']) for r in results if not pd.isna(r['seed_time'])]
        if seed_results:
            top_seed_name = min(seed_results, key=lambda x: x[1])[0]
            targets['top_seed_won'] = (winner['name'] == top_seed_name)
        else:
            targets['top_seed_won'] = np.nan
        
        return targets


    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_rows = []
        for _, row in df.iterrows():
            event_features = self.create_event_features(row)
            feature_rows.append(event_features)

        features_df = pd.DataFrame(feature_rows)

        features_df = self._add_meet_level_features(features_df)
        features_df = self._add_event_level_features(features_df)

        return features_df

    def _add_meet_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        meet_stats = df.groupby('meet').agg({
            'field_size': 'mean',
            'hhi_seed_times': 'mean',
            'pressure_index': 'mean',
            'record_holders_count': 'sum'
        }).rename(columns={
            'field_size': 'meet_avg_field_size',
            'hhi_seed_times': 'meet_avg_competitiveness',
            'pressure_index': 'meet_avg_pressure',
            'record_holders_count': 'meet_total_record_holders'
        })
        
        df = df.merge(meet_stats, left_on='meet', right_index=True, how='left')
        
        return df


    def _add_event_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['distance_category'] = pd.cut(df['distance'], 
                                       bins=[0, 100, 200, 400, 800, 2000],
                                       labels=['sprint', 'short', 'middle', 'distance', 'long_distance'])
        
        # Stroke encoding
        stroke_map = {
            'Freestyle': 'free',
            'Backstroke': 'back',
            'Breaststroke': 'breast', 
            'Butterfly': 'fly',
            'IM': 'im'
        }
        df['stroke_category'] = df['stroke'].map(stroke_map)
        
        return df


def load_and_combine_data() -> pd.DataFrame:
    data_dir = Path(__file__).parent.parent / "data" / "processed" / "clean"

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
    
    all_dfs = []
    csv_files = list(data_dir.glob("*.csv"))

    print(f"Found {len(csv_files)} CSV files in {data_dir}")

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
            print(f"Loaded {file.name} with shape {df.shape}")
        except Exception as e:
            print(f"Error loading {file.name}: {e}")

    if not all_dfs:
        raise ValueError("No valid CSV files found in the data directory.")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)

    return combined_df

def main():
    print("Engineering features")

    df = load_and_combine_data()

    feature_engineer = FeatureEngineer()

    features_df = feature_engineer.engineer_features(df)

    output_path = Path(__file__).parent.parent / "data" / "processed" / "features" / "features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features_df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")


if __name__ == "__main__":
    main()



        
        
    

        