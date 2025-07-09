import logging
import re
import pandas as pd
import pdfplumber
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from text_parser import TextParser
from data_formatter import DataFormatter

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
        self.raw_txt_dir.mkdir(parents=True, exist_ok=True)  
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging and shared parser/formatter
        utils.setup_logging()
        self.text_parser = TextParser()
        self.data_formatter = DataFormatter()
        
        # Store individual meet DataFrames in memory
        self.individual_meet_dfs = {}
    

    def parse_single_pdf(self, pdf_path: Path) -> List[Dict]:
        logging.debug(f"Parsing PDF: {pdf_path.name}")
        try:
            all_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
            
            events = self.text_parser.parse_meet_text(all_text, source_format='pdf')
            return self._process_events_for_pdf(events, pdf_path)
            
        except Exception as e:
            logging.error(f"Failed to parse {pdf_path.name}: {e}")
            return []
    

    def parse_single_txt(self, txt_path: Path) -> List[Dict]:
        logging.debug(f"Parsing TXT: {txt_path.name}")
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as file:
                all_text = file.read()
            
            # Use shared text parser with txt format
            events = self.text_parser.parse_meet_text(all_text, source_format='txt')
            return self._process_events_for_txt(events, txt_path)
            
        except Exception as e:
            logging.error(f"Failed to parse {txt_path.name}: {e}")
            return []
    

    def _process_events_for_pdf(self, events: List[Dict], pdf_path: Path) -> List[Dict]:
        meet_name = pdf_path.stem.replace('-complete-results', '').replace('-', ' ').title()
        return self._process_events(events, meet_name, pdf_path.name, pdf_path.parent.name)
    

    def _process_events_for_txt(self, events: List[Dict], txt_path: Path) -> List[Dict]:
        meet_name = txt_path.stem.replace('-complete-results', '').replace('-', ' ').title()
        return self._process_events(events, meet_name, txt_path.name, txt_path.parent.name)
    

    def _process_events(self, events: List[Dict], meet_name: str, source_file: str, meet_category: str) -> List[Dict]:
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
                'source_file': source_file,
                'meet_category': meet_category,
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
        
        return processed_events
    

    def parse_all_files(self, pdf_paths: List[Path] = None, txt_paths: List[Path] = None) -> pd.DataFrame:
        if pdf_paths is None:
            pdf_paths = list(self.raw_pdf_dir.rglob("*.pdf"))
        if txt_paths is None:
            txt_paths = list(self.raw_txt_dir.rglob("*.txt"))
        
        total_files = len(pdf_paths) + len(txt_paths)
        logging.debug(f"Parsing {len(pdf_paths)} PDF files and {len(txt_paths)} TXT files ({total_files} total)")
        
        all_events = []
        successful_parses = 0
        
        # Parse PDF files
        for pdf_path in pdf_paths:
            events = self.parse_single_pdf(pdf_path)
            if events:
                all_events.extend(events)
                successful_parses += 1
        
        # Parse TXT files
        for txt_path in txt_paths:
            events = self.parse_single_txt(txt_path)
            if events:
                all_events.extend(events)
                successful_parses += 1
        
        logging.debug(f"Successfully parsed {successful_parses}/{total_files} files")
        logging.debug(f"Total events extracted: {len(all_events)}")
        
        if not all_events:
            logging.warning("No events were successfully parsed!")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_events)
        return df
    

    def save_data(self, df: pd.DataFrame, output_path: Path) -> Path:
        if df.empty:
            logging.warning(f"No data to save for: {output_path}")
            return None
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logging.debug(f"Saved data to: {output_path}")
        return output_path


    def clean_existing_data(self) -> Tuple[Path, Path]:
        logging.info("Cleaning Existing Parsed Data")

        parsed_data_path = self.processed_dir / "parsed_events.csv"
        
        if not parsed_data_path.exists():
            logging.error(f"No parsed data found at: {parsed_data_path}")
            return None, None
        
        logging.info(f"Loading existing data from: {parsed_data_path}")
        
        try:
            original_df = pd.read_csv(parsed_data_path)
            logging.info(f"Loaded {len(original_df)} events from existing data")
            
            # Clean the data
            clean_df = self.data_formatter.clean_dataframe(original_df)
            
            # Save clean data
            clean_path = self.save_data(clean_df, self.clean_dir / "clean_events.csv")
            
            return parsed_data_path, clean_path
            
        except Exception as e:
            logging.error(f"Failed to clean existing data: {e}")
            return None, None


    def run_pipeline(self) -> Tuple[Path, Path]:
        logging.info("Starting Complete Meet Data Pipeline")
        
        logging.info("Parsing all files (PDF and TXT)")
        original_df = self.parse_all_files()
        
        logging.info("Saving processed data")
        original_path = self.save_data(original_df, self.processed_dir / "parsed_events.csv")

        logging.info("Cleaning saved data")
        clean_df = self.data_formatter.clean_dataframe(original_df)

        logging.info("Saving clean data")
        clean_path = self.save_data(clean_df, self.clean_dir / "clean_events.csv")

        return original_path, clean_path
    

def main():
    base_dir = Path(__file__).parent.parent
    output_base = base_dir / "data"
    
    pipeline = MeetDataPipeline(output_base)

    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "--parse":
            # Parse existing files only (no cleaning)
            logging.info("Running parse-only")
            df = pipeline.parse_all_files()  # This returns a DataFrame
            
            if not df.empty:  # Check if DataFrame has data
                # Save the parsed data
                parse_path = pipeline.save_data(df, pipeline.processed_dir / "parsed_events.csv")
                if parse_path:
                    print(f"\nSuccess: Generated {parse_path}")
                else:
                    print("\nFailed to save parsed data.")
            else:
                print("\nFailed to parse files - no data found.")
       
        elif command == "--clean":
            # Clean existing parsed data only
            logging.info("Running clean-only")
            parse_path, clean_path = pipeline.clean_existing_data()
            
            if clean_path:
                print(f"\nSuccess: Generated {clean_path}")
            else:
                print("\nFailed to clean existing data.")
            
    else:
        # Default behavior: run full pipeline
        parse_path, clean_path = pipeline.run_pipeline()
            
        if parse_path and clean_path:
            print(f"\nSuccess: Generated {parse_path.name}, {clean_path.name}")
        elif parse_path:
            print(f"\nPartial success: Generated {parse_path.name} only")
        else:
            print("\nPipeline failed.")


if __name__ == "__main__":
    main()
