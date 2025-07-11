import pandas as pd
import numpy as np
import ast
import re
from typing import List, Dict, Optional, Union
import logging

class DataFormatter:
    def __init__(self):
        pass


    def parse_time_to_seconds(self, time_str: str) -> Optional[float]:
        if pd.isna(time_str) or not time_str:
            return None
        
        # Handle special cases
        time_str = str(time_str).strip()
        if time_str in ['NT', 'NTX', 'X', 'DQ', 'NS', 'SCR', '--', '---']:
            return None
        
        # Remove special characters that indicate records
        time_str = re.sub(r'[#&!*bNATB]+$', '', time_str)
        
        try:
            # Handle MM:SS.HH format
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
            else:
                # Handle SS.HH format
                return float(time_str)
        except:
            return None
        

    def extract_year_from_meet(self, meet_name: str) -> Optional[int]:
        # Try to find a 4-digit year
        match = re.search(r'(\d{4})', meet_name)
        if match:
            return int(match.group(1))
        return None


    def clean_entry_dict(self, entry: Dict, time_field: str = 'prelim_time') -> Optional[Dict]:
        if not isinstance(entry, dict):
            return None
        
        # Create a copy without the 'raw' field
        cleaned = {k: v for k, v in entry.items() if k != 'raw'}
        
        # Fix school names that have times embedded in them
        if 'school' in cleaned and cleaned['school']:
            school = cleaned['school']
            # Check if there's a time pattern in the school name
            time_match = re.search(r'\s+([\d]+:[\d]+\.[\d]+|[\d]+\.[\d]+)\s*$', school)
            if time_match:
                # Extract the time and clean the school name
                cleaned['school'] = school[:time_match.start()].strip()
                # The extracted time might be the actual finals time if it's missing
                if time_field == 'finals_time' and not entry.get('finals_time'):
                    cleaned['finals_time'] = time_match.group(1)
        
        # Add time_sec for all time fields
        for field in ['seed_time', 'prelim_time', 'finals_time']:
            if field in cleaned:
                time_value = cleaned.get(field)
                cleaned[f'{field}_sec'] = self.parse_time_to_seconds(time_value)
            else:
                cleaned[f'{field}_sec'] = None
        
        # Add the main time_sec based on the appropriate time field
        time_value = entry.get(time_field)
        cleaned['time_sec'] = self.parse_time_to_seconds(time_value)
        
        # Only return if we have a valid time
        if cleaned['time_sec'] is not None:
            return cleaned
        return None


    def process_entries_list(self, entries: Union[str, List], time_field: str = 'prelim_time') -> List[Dict]:
        # Handle None or NaN values
        if entries is None:
            return []
        
        # Check for pandas NA/NaN only if it's not a list
        if not isinstance(entries, list):
            try:
                if pd.isna(entries):
                    return []
            except:
                pass
        
        # Parse string representation if needed
        if isinstance(entries, str):
            try:
                entries = ast.literal_eval(entries)
            except:
                return []
        
        if not isinstance(entries, list):
            return []
        
        # Clean each entry and filter out those without valid times
        cleaned_entries = []
        for entry in entries:
            cleaned = self.clean_entry_dict(entry, time_field)
            if cleaned:
                cleaned_entries.append(cleaned)
        
        return cleaned_entries


    def sort_entries_by_time(self, entries: List[Dict], time_field: str = 'time_sec') -> List[Dict]:
        # Add original index to preserve tie order
        for i, entry in enumerate(entries):
            entry['_original_index'] = i
        
        # Sort by the specified time field, then by original index
        # Handle None values by treating them as infinity
        def get_sort_key(entry):
            time_value = entry.get(time_field)
            if time_value is None:
                return float('inf')
            # Convert to seconds if it's a string time
            if isinstance(time_value, str):
                return self.parse_time_to_seconds(time_value) or float('inf')
            return time_value
        
        sorted_entries = sorted(entries, key=lambda x: (get_sort_key(x), x['_original_index']))
        
        # Remove the temporary index
        for entry in sorted_entries:
            del entry['_original_index']
        
        return sorted_entries


    def get_cutoff_time(self, entries: List[Dict], rank: int, time_field: str = 'time_sec') -> Optional[float]:
        if rank <= len(entries):
            return entries[rank - 1].get(time_field)
        return None


    def impute_missing_cutoff(self, cutoffs: Dict[str, Optional[float]], rank: int, 
                            all_cutoffs: List[Optional[float]]) -> Optional[float]:
        # DEBUG: Log input
        logging.debug(f"impute_missing_cutoff called for rank {rank}")
        logging.debug(f"  Current cutoffs: {cutoffs}")
        logging.debug(f"  Value for rank_{rank}: {cutoffs.get(f'rank_{rank}')}")
        
        if cutoffs[f'rank_{rank}'] is not None:
            logging.debug(f"  Returning existing value: {cutoffs[f'rank_{rank}']}")
            return cutoffs[f'rank_{rank}']
        
        # Find nearest lower and upper existing cutoffs
        lower_cutoff = None
        upper_cutoff = None
        
        # Look for lower cutoff
        for r in range(rank - 1, 0, -1):
            if r in [8, 16, 24] and cutoffs.get(f'rank_{r}') is not None:
                lower_cutoff = cutoffs[f'rank_{r}']
                logging.debug(f"  Found lower cutoff at rank {r}: {lower_cutoff}")
                break
        
        # Look for upper cutoff
        for r in range(rank + 1, 25):
            if r in [8, 16, 24] and cutoffs.get(f'rank_{r}') is not None:
                upper_cutoff = cutoffs[f'rank_{r}']
                logging.debug(f"  Found upper cutoff at rank {r}: {upper_cutoff}")
                break
        
        # Average if both exist
        if lower_cutoff is not None and upper_cutoff is not None:
            result = (lower_cutoff + upper_cutoff) / 2
            logging.debug(f"  Returning average: {result}")
            return result
        elif lower_cutoff is not None:
            logging.debug(f"  Returning lower cutoff: {lower_cutoff}")
            return lower_cutoff
        elif upper_cutoff is not None:
            logging.debug(f"  Returning upper cutoff: {upper_cutoff}")
            return upper_cutoff
        
        logging.debug(f"  Returning None")
        return None


    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Starting to clean dataframe with {len(df)} rows")

        # Filter to individual events only
        df_individual = df[df['event_type'] == 'individual'].copy()
        logging.info(f"Filtered to {len(df_individual)} individual event rows")

        df_individual = df_individual[~df_individual['distance'].isin([1000, 1650])]
        logging.info(f"Dropped 1000 and 1650 events, {len(df_individual)} rows remaining")
        
        # Extract year from meet name
        df_individual['year'] = df_individual['meet'].apply(self.extract_year_from_meet)
        
        # Create event_name from distance and stroke
        df_individual['event_name'] = df_individual.apply(
            lambda row: f"{row['distance']} {row['stroke']}", axis=1
        )
        
        # Group by meet, event, gender, distance
        grouped = df_individual.groupby(['meet', 'event_name', 'gender', 'distance', 'year', 'stroke'])
        
        results = []
        
        for group_key, group_df in grouped:
            meet, event_name, gender, distance, year, stroke = group_key
            
            # DEBUG: Log event being processed
            if event_name == "50 Backstroke" and gender == "Men" and year == 2006:
                logging.info(f"\n{'='*60}")
                logging.info(f"DEBUGGING: {year} {gender} {event_name}")
                logging.info(f"{'='*60}")
            
            # Get the first row for this group
            row = group_df.iloc[0]
            
            # Process prelims and finals
            prelims = self.process_entries_list(row['prelims'], 'prelim_time')
            finals = self.process_entries_list(row['finals'], 'finals_time')
            
            # DEBUG: Log finals count for target event
            if event_name == "50 Backstroke" and gender == "Men" and year == 2006:
                logging.info(f"Number of finals entries parsed: {len(finals)}")
                if len(finals) > 0:
                    logging.info(f"First finals entry: {finals[0].get('name')} - {finals[0].get('seed_time_sec')}")
                    logging.info(f"Last finals entry: {finals[-1].get('name')} - {finals[-1].get('seed_time_sec')}")

            # Sort finals by seed time (not finals time) for cutoff calculation
            finals_sorted_by_seed = self.sort_entries_by_time(finals, time_field='seed_time')
            if len(finals_sorted_by_seed) == 0:
                continue
            
            # DEBUG: Log sorted finals by seed time for target event
            if event_name == "50 Backstroke" and gender == "Men" and year == 2006:
                logging.info(f"\nSorted finals by seed time ({len(finals_sorted_by_seed)} entries):")
                for i, entry in enumerate(finals_sorted_by_seed[:25]):  # Show first 25
                    logging.info(f"  Rank {i+1}: {entry.get('name')} - {entry.get('seed_time_sec')}s")
            
            # Split into A, B, C entries based on seed time ranking
            A_entries = finals_sorted_by_seed[0:8]
            B_entries = finals_sorted_by_seed[8:16]
            C_entries = finals_sorted_by_seed[16:24]
            
            # Get cutoff times from seed time rankings
            cutoffs = {
                'rank_8': self.get_cutoff_time(finals_sorted_by_seed, 8, time_field='seed_time'),
                'rank_16': self.get_cutoff_time(finals_sorted_by_seed, 16, time_field='seed_time'),
                'rank_24': self.get_cutoff_time(finals_sorted_by_seed, 24, time_field='seed_time')
            }
            
            # DEBUG: Log raw cutoffs for target event
            if event_name == "50 Backstroke" and gender == "Men" and year == 2006:
                logging.info(f"\nRaw cutoffs before imputation:")
                logging.info(f"  rank_8: {cutoffs['rank_8']}")
                logging.info(f"  rank_16: {cutoffs['rank_16']}")
                logging.info(f"  rank_24: {cutoffs['rank_24']}")
                
                # Show who is at each cutoff position
                if len(finals_sorted_by_seed) >= 8:
                    logging.info(f"  8th place: {finals_sorted_by_seed[7].get('name')} - {finals_sorted_by_seed[7].get('seed_time_sec')}")
                if len(finals_sorted_by_seed) >= 16:
                    logging.info(f"  16th place: {finals_sorted_by_seed[15].get('name')} - {finals_sorted_by_seed[15].get('seed_time_sec')}")
                if len(finals_sorted_by_seed) >= 24:
                    logging.info(f"  24th place: {finals_sorted_by_seed[23].get('name')} - {finals_sorted_by_seed[23].get('seed_time_sec')}")
            
            # Impute missing cutoffs
            A_cutoff = self.impute_missing_cutoff(cutoffs, 8, list(cutoffs.values()))
            B_cutoff = self.impute_missing_cutoff(cutoffs, 16, list(cutoffs.values()))
            C_cutoff = self.impute_missing_cutoff(cutoffs, 24, list(cutoffs.values()))
            
            # DEBUG: Log final cutoffs for target event
            if event_name == "50 Backstroke" and gender == "Men" and year == 2006:
                logging.info(f"\nFinal cutoffs after imputation:")
                logging.info(f"  A_cutoff: {A_cutoff}")
                logging.info(f"  B_cutoff: {B_cutoff}")
                logging.info(f"  C_cutoff: {C_cutoff}")
                logging.info(f"{'='*60}\n")
            
            # Create result row
            result = {
                'year': year,
                'event_name': re.sub(r'\s+', '_', event_name),
                'stroke': stroke,
                'gender': gender,
                'distance': distance,
                'A_cutoff_sec': A_cutoff,
                'B_cutoff_sec': B_cutoff,
                'C_cutoff_sec': C_cutoff,
                'A_entries': A_entries,
                'B_entries': B_entries,
                'C_entries': C_entries,
                'finals': finals,
                'prelims': self.sort_entries_by_time(prelims, 'prelim_time_sec')  # Sort prelims by prelim time
            }
            
            results.append(result)
        
        # Create final dataframe
        clean_df = pd.DataFrame(results)
        
        # Sort by year, event_name, gender
        clean_df = clean_df.sort_values(['year', 'event_name', 'gender'])
        
        logging.info(f"Created dataframe with {len(clean_df)} rows")
        
        return clean_df