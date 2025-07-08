import os
import time
import logging
import re
import pandas as pd
import pdfplumber
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
import ast


import config
import utils


class MeetDataPipeline:
    def __init__(self, output_base: Path):
        self.output_base = output_base
        self.raw_pdf_dir = output_base / "raw" / "pdfs"
        self.raw_txt_dir = output_base / "raw" / "txts"
        self.processed_dir = output_base / "processed" / "parsed"
        self.clean_dir = output_base / "processed" / "clean"
        
        # Create directories
        self.raw_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        utils.setup_logging()
    

    def parse_meet_text(self, text: str) -> List[Dict]:
        """Parse meet text with improved event state management."""
        
        # Event tracking dictionary - key: (event_num, gender, distance, stroke)
        events_dict = {}
        
        # Regex patterns
        event_re = re.compile(r'^Event\s+(\d+)\s+(Women|Men)\s+(\d+)\s+Yard\s+([A-Za-z ]+)(?:\s+Time\s+Trial)?$')
        
        # IMPROVED REGEX: Properly handles "Last, First" swimmer names and relay teams
        individual_entry_re = re.compile(
            r'^(\*?\d+|---)\s+([A-Za-z\',.-]+,\s+[A-Za-z\',.-]+)\s+([A-Z]{2})\s+([A-Za-z ]+(?:[A-Za-z])+)\s+([\d:.NTXb]+)\s+([\d:.NTXb]+)(?:\s+([\d:.NTXb]+))?(?:\s+(\d+))?'
        )
        
        # Relay team regex (e.g., "1 Hamilton College A NT 1:34.44")
        relay_entry_re = re.compile(
            r'^(\*?\d+|---)\s+([A-Za-z ]+)\s+([A-Z])\s+([\d:.NTXb]+)\s+([\d:.NTXb]+)(?:\s+([\d:.NTXb]+))?(?:\s+(\d+))?'
        )

        diving_re = re.compile(r'^Event\s+(\d+)\s+(Women|Men)\s+([13])\s+mtr\s+Diving')
        
        # State variables
        current_event_key = None
        current_section = None
        
        def get_event_key(event_line: str) -> Optional[Tuple[str, str, int, str]]:
            """Extract normalized event key from event header line."""
            match = event_re.match(event_line)
            if match:
                event_num, gender, distance, stroke = match.groups()
                # Skip Time Trial events
                if 'time trial' in event_line.lower():
                    #logging.debug(f"Skipping Time Trial event: {event_line}")
                    return None
                if is_diving_event(event_line):
                    #logging.debug(f"Skipping Diving event: {event_line}")
                    return None
                return (event_num, gender, int(distance), stroke.strip())
            return None
        
        def is_diving_event(event_line: str) -> bool:
            return diving_re.match(event_line)
        
        def is_any_skipped_event(event_line: str) -> bool:
            """Check if this is any type of event we want to skip."""
            return (is_diving_event(event_line) or 
                    'time trial' in event_line.lower() or
                    'swim-off' in event_line.lower())

        def ensure_event_exists(event_key: Tuple, event_line: str):
            """Ensure event exists in events_dict, create if necessary."""
            if event_key not in events_dict:
                # Determine if this is a relay event
                is_relay = any(word in event_line.lower() for word in ['relay', 'medley relay', 'freestyle relay'])
                
                if is_relay:
                    events_dict[event_key] = {
                        'event': event_line,
                        'results': [],  # Relays use 'results' instead of finals/prelims
                        'event_type': 'relay'
                    }
                else:
                    events_dict[event_key] = {
                        'event': event_line,
                        'finals': [],
                        'prelims': [],
                        'event_type': 'individual'
                    }
                #logging.debug(f"Created new {'relay' if is_relay else 'individual'} event: {event_line}")
            #else:
                #logging.debug(f"Continuing existing event: {event_line}")
        
        def is_section_header(line: str) -> Optional[str]:
            """Check if line is a section header and return section type."""
            line_lower = line.lower()
            if line_lower.startswith('finals') or 'final' in line_lower:
                return 'finals'
            elif line_lower.startswith('prelim'):
                return 'prelims'
            return None
        
        def parse_entry(line: str) -> Optional[Dict]:
            """Parse a swimmer/relay entry line."""
            # Skip exhibition swims (entries with --- rank or X times)
            if line.startswith('---') or ' X' in line or line.endswith(' X'):
                #logging.debug(f"Skipping exhibition swim: {line}")
                return None
            
            # Try individual swimmer first
            m = individual_entry_re.match(line)
            if m:
                entry = {
                    'raw': line,
                    'rank': m.group(1),
                    'name': m.group(2),
                    'yr': m.group(3),
                    'school': m.group(4),
                    'entry_type': 'individual'
                }
                
                # Handle time assignments based on current section
                if current_section == 'finals':
                    # In finals: first time = prelim time, second time = finals time
                    entry['seed_time'] = None  # No seed time in finals results
                    entry['prelim_time'] = m.group(5)  # This is their prelim time
                    entry['finals_time'] = m.group(6)  # This is their finals time
                    entry['points'] = m.group(8)
                else:
                    # In prelims: first time = seed time, second time = prelim time
                    entry['seed_time'] = m.group(5)
                    entry['prelim_time'] = m.group(6)
                    entry['finals_time'] = m.group(7)  # Usually None in prelims
                    entry['points'] = m.group(8)
                
                return entry
        
        # Main parsing loop
        lines = text.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if this is ANY event header (including ones we want to skip)
            if line.startswith('Event'):
                # Check if we should skip this event
                if is_any_skipped_event(line):
                    #logging.debug(f"Skipping event and clearing context: {line}")
                    current_event_key = None  # CRITICAL: Clear the current context
                    current_section = None
                    continue
                
                # Check if this is a valid swimming event
                event_key = get_event_key(line)
                if event_key:
                    current_event_key = event_key
                    ensure_event_exists(event_key, line)
                    current_section = None  # Reset section when new event found
                    continue
                else:
                    # If we can't parse it as a valid swimming event, clear context
                    #logging.debug(f"Failed to parse event header, clearing context: {line}")
                    current_event_key = None
                    current_section = None
                    continue
            
            # Check for section header (only process if we have a valid current event)
            if current_event_key:
                section = is_section_header(line)
                if section:
                    current_section = section
                    #logging.debug(f"Section changed to {section.upper()} for event: {events_dict[current_event_key]['event']}")
                    continue
            
            # Try to parse entry (only if we have a valid current event and section)
            if current_event_key and current_section:
                entry = parse_entry(line)
                if entry:
                    # Handle relay vs individual events differently
                    if events_dict[current_event_key]['event_type'] == 'relay':
                        # For relays, ignore section and add to 'results'
                        events_dict[current_event_key]['results'].append(entry)
                        #logging.debug(f"Adding {entry['entry_type'].upper()} entry to RELAY results: {entry['name']}")
                    else:
                        # For individual events, add to appropriate section
                        events_dict[current_event_key][current_section].append(entry)
                        #logging.debug(f"Adding {entry['entry_type'].upper()} entry to {current_section.upper()}: {entry['name']}")
                    continue
            
            # Log unmatched lines for debugging (but only if we have a current event context)
            #if (current_event_key and line and 
                #not line.startswith('===') and not line.startswith('---') and
                #not line.startswith('Seed') and not line.startswith('Name') and
                #not line.startswith('r:+') and  # Reaction time lines
                #len(line) > 3):
                #logging.debug(f"Failed to parse line: {line}")
        
        # Convert events_dict to list format
        events = list(events_dict.values())
        
        # Print summary of parsed events
        logging.info(f"Parsed events summary:")
        total_events = len(events)
        individual_events = [e for e in events if e.get('event_type') == 'individual']
        relay_events = [e for e in events if e.get('event_type') == 'relay']
        
        total_finals = sum(len(e.get('finals', [])) for e in individual_events)
        total_prelims = sum(len(e.get('prelims', [])) for e in individual_events)
        total_relay_results = sum(len(e.get('results', [])) for e in relay_events)
        
        logging.info(f"  Total unique events: {total_events}")
        logging.info(f"  Individual events: {len(individual_events)} (Finals: {total_finals}, Prelims: {total_prelims})")
        logging.info(f"  Relay events: {len(relay_events)} (Results: {total_relay_results})")
        
        for e in events:
            if e.get('event_type') == 'relay':
                #logging.debug(f"RELAY Event: {e['event']} | Results: {len(e.get('results', []))}")
                continue
            else:
                logging.debug(f"INDIVIDUAL Event: {e['event']} | Finals: {len(e.get('finals', []))} | Prelims: {len(e.get('prelims', []))}")
        
        return events
    

    def parse_single_pdf(self, pdf_path: Path) -> List[Dict]:
        logging.info(f"Parsing PDF: {pdf_path.name}")
        try:
            all_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
            
            events = self.parse_meet_text(all_text)
            meet_name = pdf_path.stem.replace('-complete-results', '').replace('-', ' ').title()
            processed_events = []
            
            for event in events:
                event_match = re.search(r'Event\s+(\d+)\s+(Women|Men)\s+(\d+)\s+Yard\s+([A-Za-z ]+)', event['event'])
                if not event_match:
                    continue
                event_num, gender, distance, stroke = event_match.groups()
                
                # Base event info
                processed_event = {
                    'event': event_num,
                    'meet': meet_name,
                    'stroke': stroke.strip(),
                    'gender': gender,
                    'distance': int(distance),
                    'source_file': pdf_path.name,
                    'meet_category': pdf_path.parent.name,
                    'event_type': event.get('event_type', 'individual')
                }
                
                # Add appropriate results based on event type
                if event.get('event_type') == 'relay':
                    processed_event['results'] = event.get('results', [])
                    processed_event['finals'] = []  # Empty for compatibility
                    processed_event['prelims'] = []  # Empty for compatibility
                else:
                    processed_event['finals'] = event.get('finals', [])
                    processed_event['prelims'] = event.get('prelims', [])
                    processed_event['results'] = []  # Empty for compatibility
                
                processed_events.append(processed_event)
            
            # Count individual vs relay results
            individual_events = [e for e in processed_events if e.get('event_type') == 'individual']
            relay_events = [e for e in processed_events if e.get('event_type') == 'relay']
            
            total_individual = sum(len(e.get('finals', [])) + len(e.get('prelims', [])) for e in individual_events)
            total_relay = sum(len(e.get('results', [])) for e in relay_events)
            
            logging.info(f"  Extracted {len(processed_events)} events ({len(individual_events)} individual, {len(relay_events)} relay)")
            logging.info(f"  Individual results: {total_individual}, Relay results: {total_relay}")
            return processed_events
        except Exception as e:
            logging.error(f"Failed to parse {pdf_path.name}: {e}")
            return []
    

    def extract_time_predictions_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract swimmer times, entry times, prelim results, and finals results
        for time cutoff prediction analysis.
        """
        logging.info("Extracting time prediction data...")
        
        prediction_records = []
        
        for _, event_row in df.iterrows():
            # Skip relay events for prediction analysis
            if event_row.get('event_type') == 'relay':
                continue
                
            event_info = {
                'event_num': event_row['event'],
                'meet': event_row['meet'],
                'stroke': event_row['stroke'],
                'gender': event_row['gender'],
                'distance': event_row['distance'],
                'meet_category': event_row['meet_category'],
                'source_file': event_row['source_file']
            }
            
            # Track which swimmers made finals
            finalists = set()
            if isinstance(event_row['finals'], list):
                for result in event_row['finals']:
                    if result.get('entry_type') == 'individual':
                        finalists.add(result['name'])
            
            # Process all swimmers from prelims (includes everyone)
            if isinstance(event_row['prelims'], list):
                for result in event_row['prelims']:
                    if result.get('entry_type') == 'individual':
                        made_finals = result['name'] in finalists
                        
                        record = {
                            **event_info,
                            'swimmer_name': result['name'],
                            'class_year': result['yr'],
                            'school': result['school'],
                            'seed_time': result['seed_time'],
                            'prelim_time': result['prelim_time'],
                            'prelim_rank': result['rank'],
                            'made_finals': made_finals,
                            'entry_type': 'individual'
                        }
                        
                        # Add finals info if they made it
                        if made_finals:
                            # Find their finals result
                            for finals_result in event_row['finals']:
                                if (finals_result.get('entry_type') == 'individual' and 
                                    finals_result['name'] == result['name']):
                                    record['finals_time'] = finals_result.get('prelim_time')  # This is actually finals time in the data
                                    record['final_rank'] = finals_result['rank']
                                    record['points'] = finals_result.get('points')
                                    break
                        
                        prediction_records.append(record)
            
            # Handle events that only have finals (no prelims)
            elif isinstance(event_row['finals'], list) and len(event_row['finals']) > 0:
                for result in event_row['finals']:
                    if result.get('entry_type') == 'individual':
                        record = {
                            **event_info,
                            'swimmer_name': result['name'],
                            'class_year': result['yr'],
                            'school': result['school'],
                            'seed_time': result['seed_time'],
                            'finals_time': result.get('prelim_time'),  # This is actually finals time
                            'final_rank': result['rank'],
                            'points': result.get('points'),
                            'made_finals': True,
                            'entry_type': 'individual'
                        }
                        prediction_records.append(record)
        
        prediction_df = pd.DataFrame(prediction_records)
        
        if not prediction_df.empty:
            logging.info(f"Extracted {len(prediction_df)} individual swimmer results for prediction analysis")
            logging.info(f"  - Results that made finals: {len(prediction_df[prediction_df['made_finals'] == True])}")
            logging.info(f"  - Prelim-only results: {len(prediction_df[prediction_df['made_finals'] == False])}")
        
        return prediction_df
    

    def parse_all_pdfs(self, pdf_paths: List[Path]) -> pd.DataFrame:
        logging.info(f"Parsing {len(pdf_paths)} PDF files...")
        
        all_events = []
        successful_parses = 0
        
        for pdf_path in pdf_paths:
            events = self.parse_single_pdf(pdf_path)
            if events:
                all_events.extend(events)
                successful_parses += 1
        
        logging.info(f"Successfully parsed {successful_parses}/{len(pdf_paths)} PDFs")
        logging.info(f"Total events extracted: {len(all_events)}")
        
        if not all_events:
            logging.warning("No events were successfully parsed!")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_events)
        
        # Add summary statistics
        individual_events = df[df.get('event_type', 'individual') == 'individual']
        relay_events = df[df.get('event_type', 'individual') == 'relay']
        
        events_with_results = len(individual_events[(individual_events['finals'].apply(len) > 0) | 
                                                   (individual_events['prelims'].apply(len) > 0)]) + \
                             len(relay_events[relay_events['results'].apply(len) > 0])
        
        total_race_results = (sum(len(f) + len(p) for f, p in zip(individual_events['finals'], individual_events['prelims'])) +
                             sum(len(r) for r in relay_events['results']))
        
        logging.info(f"Events with race results: {events_with_results}/{len(df)}")
        logging.info(f"Total individual race results: {total_race_results}")
        
        return df

    def deduplicate_records(self, records: List[Dict]) -> List[Dict]:
        seen_records: Set[Tuple] = set()
        unique_records = []
        
        for record in records:
            # Create a tuple key for deduplication
            record_key = (
                record['type'],
                record['time'],
                record['date'],
                record['athlete']
            )
            
            if record_key not in seen_records:
                seen_records.add(record_key)
                unique_records.append(record)
        
        return unique_records

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Cleaning DataFrame")
        if df.empty:
            return df
        def has_any_entries(col):
            if pd.isna(col):
                return False
            try:
                entries = ast.literal_eval(str(col))
                return len(entries) > 0
            except Exception:
                return False
        
        # For individual events: Keep rows where either finals or prelims has at least one entry
        # For relay events: Keep rows where results has at least one entry
        keep_mask = pd.Series([False] * len(df))
        
        for idx, row in df.iterrows():
            if row.get('event_type') == 'relay':
                keep_mask.iloc[idx] = has_any_entries(row.get('results', []))
            else:
                has_finals = has_any_entries(row.get('finals', []))
                has_prelims = has_any_entries(row.get('prelims', []))
                keep_mask.iloc[idx] = has_finals or has_prelims
        
        cleaned_df = df[keep_mask].copy()
        original_count = len(df)
        final_count = len(cleaned_df)
        removed_count = original_count - final_count
        logging.info(f"Removed {removed_count} rows with no entries")
        return cleaned_df.reset_index(drop=True)

    def save_processed_data(self, df: pd.DataFrame) -> Path:
        if df.empty:
            logging.warning("No data exists")
            return None
        
        output_path = self.processed_dir / "parsed_events.csv"
        df.to_csv(output_path, index=False)
        
        # Debug feature: Save individual meets breakdown
        individual_meets_dir = self.processed_dir / "individual_meets"
        individual_meets_dir.mkdir(exist_ok=True)
        
        if not df.empty and 'meet' in df.columns:
            for meet_name in df['meet'].unique():
                meet_df = df[df['meet'] == meet_name]
                safe_meet_name = meet_name.replace(' ', '_').replace('/', '_')
                meet_path = individual_meets_dir / f"{safe_meet_name}.csv"
                meet_df.to_csv(meet_path, index=False)
                logging.debug(f"Saved individual meet data: {meet_path}")
        
        logging.info(f"Saved processed data to: {output_path}")
        
        return output_path
    
    def save_clean_data(self, df: pd.DataFrame) -> Path:
        if df.empty:
            logging.warning("No data exists")
            return None
        
        output_path = self.clean_dir / "clean_events.csv"
        df.to_csv(output_path, index=False)
        
        logging.info(f"Saved clean data to: {output_path}")
        
        return output_path
    
    def save_prediction_data(self, df: pd.DataFrame) -> Path:
        """Save the extracted prediction data for time cutoff analysis."""
        if df.empty:
            logging.warning("No prediction data exists")
            return None
        
        output_path = self.clean_dir / "prediction_data.csv"
        df.to_csv(output_path, index=False)
        
        logging.info(f"Saved prediction data to: {output_path}")
        
        return output_path

    def run_pipeline(self) -> Tuple[Path, Path, Path]:
        logging.info("Starting Complete Meet Data Pipeline")
        
        pdf_paths = list(self.raw_pdf_dir.rglob("*.pdf"))

        logging.info("Parsing PDFs")
        original_df = self.parse_all_pdfs(pdf_paths)
        
        logging.info("Saving processed data")
        original_path = self.save_processed_data(original_df)

        logging.info("Cleaning saved data")
        clean_df = self.clean_dataframe(original_df)

        logging.info("Saving clean data")
        clean_path = self.save_clean_data(clean_df)
        
        logging.info("Extracting prediction data")
        prediction_df = self.extract_time_predictions_data(clean_df)
        
        logging.info("Saving prediction data")
        prediction_path = self.save_prediction_data(prediction_df)

        return original_path, clean_path, prediction_path

    def parse_existing_pdfs(self) -> Path:
        logging.info("Parsing Existing PDFs")
        
        pdf_paths = list(self.raw_pdf_dir.rglob("*.pdf"))
        
        if not pdf_paths:
            logging.warning(f"No PDFs found in {self.raw_pdf_dir}")
            return None
        
        logging.info(f"Found {len(pdf_paths)} existing PDFs")

        df = self.parse_all_pdfs(pdf_paths)
        output_path = self.save_processed_data(df)
        
        return output_path

    def clean_existing_data(self) -> Tuple[Path, Path, Path]:
        logging.info("Cleaning Existing Parsed Data")
    
        parsed_data_path = self.processed_dir / "parsed_events.csv"
        
        if not parsed_data_path.exists():
            logging.error(f"No parsed data found at: {parsed_data_path}")
            return None, None, None
        
        logging.info(f"Loading existing data from: {parsed_data_path}")
        
        try:
            original_df = pd.read_csv(parsed_data_path)
            logging.info(f"Loaded {len(original_df)} events from existing data")
            
            # Clean the data
            clean_df = self.clean_dataframe(original_df)
            
            # Extract prediction data
            prediction_df = self.extract_time_predictions_data(clean_df)
            
            # Save data
            clean_path = self.save_clean_data(clean_df)
            prediction_path = self.save_prediction_data(prediction_df)
            
            return parsed_data_path, clean_path, prediction_path
            
        except Exception as e:
            logging.error(f"Failed to clean existing data: {e}")
            return None, None, None


def main():
    base_dir = Path(__file__).parent.parent
    output_base = base_dir / "data"
    
    pipeline = MeetDataPipeline(output_base)

    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--parse" and sys.argv[2] == "--clean":
        # Parse existing PDFs and then clean the data
        parse_path = pipeline.parse_existing_pdfs()
        if parse_path:
            _, clean_path, prediction_path = pipeline.clean_existing_data()
        else:
            clean_path = None
            prediction_path = None
    elif len(sys.argv) > 1 and sys.argv[1] == "--parse":
        # Parse existing PDFs only
        parse_path = pipeline.parse_existing_pdfs()
        clean_path = None
        prediction_path = None
    elif len(sys.argv) > 1 and sys.argv[1] == "--clean":
        # Clean existing parsed data only
        parse_path, clean_path, prediction_path = pipeline.clean_existing_data()
    else:
        parse_path, clean_path, prediction_path = pipeline.run_pipeline()
    
    # Print results
    if parse_path and clean_path and prediction_path:
        print(f"\n✅ Success: Generated {parse_path.name}, {clean_path.name}, and {prediction_path.name}")
    elif parse_path and clean_path:
        print(f"\n✅ Success: Generated {parse_path.name} and {clean_path.name}")
    elif parse_path:
        print(f"\n✅ Success: Generated {parse_path.name}")
    else:
        print("\n❌ Failed to generate any output files.")


if __name__ == "__main__":
    main()