import os
import time
import logging
import re
import pandas as pd
import pdfplumber
from pathlib import Path
from typing import List, Dict, Tuple, Set
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
        # Extract year from header
        year_match = re.search(r"(\d{4})\s+Mens? NESCAC Championship", text)
        meet_year = year_match.group(1) if year_match else None

        events = []
        event_re = re.compile(r'^Event\s+(\d+)\s+(Women|Men)\s+(\d+)\s+Yard\s+([A-Za-z ]+)')
        record_re = re.compile(r'^(NESCAC|MEET|Pool|NCAA \'[AB]\'|\d{4}-it took):\s+([\d:.]+)([A-Z]?)\s*(.*)')
        entry_re = re.compile(
            r'^(\*?\d+|---)\s+([A-Za-z\',.-]+)\s+([A-Za-z]+)\s+([A-Za-z ]+)\s+([\d:.NTX]+)\s+([\d:.NTX]+)(?:\s+([\d:.NTX]+))?(?:\s+(\d+))?')
        # Name Yr School Seed Time Prelim Time Finals Time Points
        # or Name Yr School Seed Time Prelim Time

        current_event = None
        current_records = []
        current_finals = []
        current_prelims = []
        section = None

        def flush_event():
            if current_event:
                events.append({
                    'event': current_event,
                    'year': meet_year,
                    'records': current_records.copy(),
                    'finals': current_finals.copy(),
                    'prelims': current_prelims.copy(),
                })

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Detect new event
            if line.startswith('Event '):
                flush_event()
                current_event = line
                current_records = []
                current_finals = []
                current_prelims = []
                section = None
                continue
            # Detect section headers
            if line.lower().startswith('finals') or 'final' in line.lower():
                section = 'finals'
                continue
            if line.lower().startswith('prelim'):
                section = 'prelims'
                continue
            # Detect records
            rec_match = record_re.match(line)
            if rec_match:
                current_records.append({
                    'type': rec_match.group(1),
                    'time': rec_match.group(2),
                    'code': rec_match.group(3),
                    'extra': rec_match.group(4)
                })
                continue
            # Detect swimmer entry
            m = entry_re.match(line)
            if m and section:
                entry = {
                    'raw': line,
                    'rank': m.group(1),
                    'name': m.group(2),
                    'yr': m.group(3),
                    'school': m.group(4),
                    'seed_time': m.group(5),
                    'prelim_time': m.group(6),
                    'finals_time': m.group(7),
                    'points': m.group(8)
                }
                if section == 'finals':
                    current_finals.append(entry)
                elif section == 'prelims':
                    current_prelims.append(entry)
                continue
        # Flush last event
        flush_event()
        return events
    

    def parse_single_pdf(self, pdf_path: Path) -> List[Dict]:
        logging.info(f"Parsing PDF: {pdf_path.name}")
        
        try:
            # Extract text from PDF
            all_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"

            #logging.info(all_text)
            
            # Parse events using exact same logic
            events = self.parse_meet_text(all_text)
            
            # Extract meet name from filename
            meet_name = pdf_path.stem.replace('-complete-results', '').replace('-', ' ').title()
            
            # Add meet context to each event
            processed_events = []
            for event in events:
                # Extract event details
                event_match = re.search(r'Event\s+(\d+)\s+(Women|Men)\s+(\d+)\s+Yard\s+([A-Za-z ]+)', event['event'])
                if not event_match:
                    continue
                    
                event_num, gender, distance, stroke = event_match.groups()
                
                processed_event = {
                    'event': event_num,
                    'meet': meet_name,
                    'stroke': stroke.strip(),
                    'gender': gender,
                    'distance': int(distance),
                    'source_file': pdf_path.name,
                    'meet_category': pdf_path.parent.name,
                    'records': [
                        [record.get('type'), record.get('time'), record.get('code'), record.get('extra')] for record in event['records']
                    ],
                    'finals': event['finals'],
                    'prelims': event['prelims']
                }
                processed_events.append(processed_event)
            
            logging.info(f"  Extracted {len(processed_events)} events, {sum(len(e.get('finals', [])) + len(e.get('prelims', [])) for e in processed_events)} total results")
            return processed_events
            
        except Exception as e:
            logging.error(f"Failed to parse {pdf_path.name}: {e}")
            return []
    

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
        events_with_results = len(df[df['entries'].apply(len) > 0])
        total_race_results = sum(len(entries) for entries in df['entries'])
        
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
        
        def has_entries(entries_str):
            if pd.isna(entries_str):
                return False
            try:
                entries = ast.literal_eval(str(entries_str))
                return len(entries) > 0
            except:
                return False
        
        def has_records(records_str):
            if pd.isna(records_str):
                return False
            try:
                records = ast.literal_eval(str(records_str))
                return len(records) > 0
            except:
                return False
            
        has_entries_mask = df['entries'].apply(has_entries)
        has_records_mask = df['records'].apply(has_records)

        keep_mask = has_entries_mask & has_records_mask

        cleaned_df = df[keep_mask].copy()

        original_count = len(df)
        final_count = len(cleaned_df)
        removed_count = original_count - final_count

        logging.info(f"Removed {removed_count} rows with no entries or records")

        return cleaned_df.reset_index(drop=True)


    def save_processed_data(self, df: pd.DataFrame) -> Path:
        if df.empty:
            logging.warning("No data exists")
            return None
        
        output_path = self.processed_dir / "parsed_events.csv"
        df.to_csv(output_path, index=False)
        
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
    

    def run_pipeline(self) -> Tuple[Path, Path]:
        logging.info("Starting Complete Meet Data Pipeline")
        
        pdf_paths = list(self.raw_pdf_dir.rglob("*.pdf"))

        logging.info("Parsing PDFs")
        original_df = self.parse_all_pdfs(pdf_paths)
        
        logging.info("Saving processed data")
        original_path = self.save_processed_data(original_df)

        logging.info("Cleaning saved data")
        clean_df = self.clean_dataframe(original_df)

        logging.info("Saving clean data")
        self.save_clean_data(clean_df)

        return original_path, clean_df
    

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


    def clean_existing_data(self) -> Tuple[Path, Path]:
        logging.info("Cleaning Existing Parsed Data")
    
        parsed_data_path = self.processed_dir / "parsed_events.csv"
        
        if not parsed_data_path.exists():
            logging.info(f"No parsed data found at: {parsed_data_path}", "ERROR")
            return None, None
        
        logging.info(f"Loading existing data from: {parsed_data_path}")
        
        try:
            original_df = pd.read_csv(parsed_data_path)
            logging.info(f"Loaded {len(original_df)} events from existing data")
            
            # Clean the data
            clean_df = self.clean_dataframe(original_df)
            
            # Save data
            clean_path = self.save_clean_data(clean_df)
            
            return parsed_data_path, clean_path
            
        except Exception as e:
            logging.info(f"Failed to clean existing data: {e}", "ERROR")
            return None, None


def main():
    base_dir = Path(__file__).parent.parent
    output_base = base_dir / "data"
    
    pipeline = MeetDataPipeline(output_base)

    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--parse" and sys.argv[2] == "--clean":
        # Parse existing PDFs and then clean the data
        parse_path = pipeline.parse_existing_pdfs()
        if parse_path:
            _, clean_path = pipeline.clean_existing_data()
        else:
            clean_path = None
    elif len(sys.argv) > 1 and sys.argv[1] == "--parse":
        # Parse existing PDFs only
        parse_path = pipeline.parse_existing_pdfs()
    elif len(sys.argv) > 1 and sys.argv[1] == "--clean":
        # Clean existing parsed data only
        parse_path, clean_path = pipeline.clean_existing_data()
    else:
        parse_path, clean_path = pipeline.run_pipeline()
    
    if parse_path and clean_path:
        print(f"\n Success: {parse_path.name} and {clean_path.name}")
    elif parse_path and not clean_path:
        print(f"\n Success: {parse_path.name} (no clean data generated)")
    else:
        print("\nFailed.")


if __name__ == "__main__":
    main()