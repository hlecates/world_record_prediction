import os
import time
import logging
import re
import pandas as pd
import pdfplumber
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union
import ast

import config
import utils

class TextParser:
    """Handles parsing of swimming meet text data from both PDF extracts and direct text files."""
    
    def __init__(self):
        # Shared regex patterns
        self.event_re = re.compile(r'^Event\s+(\d+)\s+(Women|Men)\s+(\d+)\s+Yard\s+([A-Za-z ]+)(?:\s+Time\s+Trial)?$')
        self.individual_entry_re = re.compile(
            r'^(\*?\d+|---)\s+([A-Za-z\',.-]+,\s+[A-Za-z\',.-]+)\s+([A-Z]{2})\s+([A-Za-z ]+(?:[A-Za-z])+)\s+([\d:.NTXb]+)\s+([\d:.NTXb]+)(?:\s+([\d:.NTXb]+))?(?:\s+(\d+))?'
        )
        self.relay_entry_re = re.compile(
            r'^(\*?\d+|---)\s+([A-Za-z ]+)\s+([A-Z])\s+([\d:.NTXb]+)\s+([\d:.NTXb]+)(?:\s+([\d:.NTXb]+))?(?:\s+(\d+))?'
        )
        self.diving_re = re.compile(r'^Event\s+(\d+)\s+(Women|Men)\s+([13])\s+mtr\s+Diving')
    
    def preprocess_text(self, text: str, source_format: str = 'auto') -> str:
        """
        Preprocess text based on source format.
        
        Args:
            text: Raw text content
            source_format: 'pdf', 'txt', or 'auto'
        """
        if source_format == 'auto':
            source_format = self._detect_format(text)
        
        if source_format == 'txt':
            return self._preprocess_txt_format(text)
        else:  # pdf or unknown
            return self._preprocess_pdf_format(text)
    
    def _detect_format(self, text: str) -> str:
        """Auto-detect if text came from PDF extraction or direct text file."""
        # Look for PDF-specific artifacts
        pdf_indicators = [
            'HY-TEK\'S MEET MANAGER',  # Common in your txt files from PDF
            'Licensed to',             # Another PDF indicator
            'COMPLETE RESULTS'         # Header format
        ]
        
        if any(indicator in text for indicator in pdf_indicators):
            return 'txt'  # Actually from PDF but saved as txt
        
        # Look for cleaner formatting that suggests direct text
        lines = text.split('\n')
        clean_event_lines = [line for line in lines if line.strip().startswith('Event')]
        
        if len(clean_event_lines) > 0:
            # Check if events are cleanly formatted
            return 'txt'
        
        return 'pdf'  # Default to PDF processing
    
    def _preprocess_txt_format(self, text: str) -> str:
        """
        Preprocess text files that may have different formatting.
        Handle issues like:
        - Extra spacing
        - Different line endings
        - Encoding issues
        """
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            # Strip extra whitespace but preserve structure
            line = line.rstrip()
            
            # Skip empty lines and header cruft
            if not line.strip():
                continue
            
            # Skip common header lines that aren't useful
            skip_patterns = [
                r'^Licensed to',
                r'^HY-TEK\'S MEET MANAGER',
                r'^COMPLETE RESULTS$',
                r'^\s*Results\s*$',
                r'^\d{4}.*Championship.*\d{4}$',  # Date ranges
                r'^={20,}$',  # Long equal signs
                r'^-{20,}$'   # Long dashes
            ]
            
            if any(re.match(pattern, line) for pattern in skip_patterns):
                continue
            
            processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def _preprocess_pdf_format(self, text: str) -> str:
        """
        Preprocess PDF-extracted text.
        Handle PDF-specific issues like:
        - Inconsistent spacing
        - Line breaks in wrong places
        - OCR artifacts
        """
        # For now, minimal preprocessing since your PDF extraction seems clean
        return text
    
    def parse_meet_text(self, text: str, source_format: str = 'auto') -> List[Dict]:
        """
        Main parsing method that works for both PDF and text sources.
        
        Args:
            text: Raw text content
            source_format: 'pdf', 'txt', or 'auto'
        """
        # Preprocess based on format
        processed_text = self.preprocess_text(text, source_format)
        
        # Use your existing parsing logic with the processed text
        return self._parse_processed_text(processed_text)
    
    def _parse_processed_text(self, text: str) -> List[Dict]:
        """
        Core parsing logic - identical to your existing parse_meet_text method.
        This is the shared logic that works for both PDF and txt sources.
        """
        # Event tracking dictionary - key: (event_num, gender, distance, stroke)
        events_dict = {}
        
        # State variables
        current_event_key = None
        current_section = None
        
        def get_event_key(event_line: str) -> Optional[Tuple[str, str, int, str]]:
            """Extract normalized event key from event header line."""
            match = self.event_re.match(event_line)
            if match:
                event_num, gender, distance, stroke = match.groups()
                # Skip Time Trial events
                if 'time trial' in event_line.lower():
                    return None
                if self.is_diving_event(event_line):
                    return None
                return (event_num, gender, int(distance), stroke.strip())
            return None
        
        def is_any_skipped_event(event_line: str) -> bool:
            """Check if this is any type of event we want to skip."""
            return (self.is_diving_event(event_line) or 
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
                return None
            
            # Try individual swimmer first
            m = self.individual_entry_re.match(line)
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
                    entry['seed_time'] = None
                    entry['prelim_time'] = m.group(5)
                    entry['finals_time'] = m.group(6)
                    entry['points'] = m.group(8)
                else:
                    entry['seed_time'] = m.group(5)
                    entry['prelim_time'] = m.group(6)
                    entry['finals_time'] = m.group(7)
                    entry['points'] = m.group(8)
                
                return entry
            
            # Try relay team (reuse existing logic)
            return None
        
        # Main parsing loop - identical to your existing logic
        lines = text.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if this is ANY event header (including ones we want to skip)
            if line.startswith('Event'):
                # Check if we should skip this event
                if is_any_skipped_event(line):
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
                    current_event_key = None
                    current_section = None
                    continue
            
            # Check for section header (only process if we have a valid current event)
            if current_event_key:
                section = is_section_header(line)
                if section:
                    current_section = section
                    continue
            
            # Try to parse entry (only if we have a valid current event and section)
            if current_event_key and current_section:
                entry = parse_entry(line)
                if entry:
                    # Handle relay vs individual events differently
                    if events_dict[current_event_key]['event_type'] == 'relay':
                        events_dict[current_event_key]['results'].append(entry)
                    else:
                        events_dict[current_event_key][current_section].append(entry)
                    continue
        
        # Convert events_dict to list format and return
        events = list(events_dict.values())
        
        # Log summary
        logging.info(f"Parsed {len(events)} events from text")
        
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
                logging.debug(f"RELAY Event: {e['event']} | Results: {len(e.get('results', []))}")
            else:
                logging.debug(f"INDIVIDUAL Event: {e['event']} | Finals: {len(e.get('finals', []))} | Prelims: {len(e.get('prelims', []))}")
        
        return events
    
    def is_diving_event(self, event_line: str) -> bool:
        """Check if event is a diving event."""
        return self.diving_re.match(event_line) is not None


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
        
        # Setup logging and shared parser
        utils.setup_logging()
        self.text_parser = TextParser()
        
        # Store individual meet DataFrames in memory
        self.individual_meet_dfs = {}
    
    def parse_single_pdf(self, pdf_path: Path) -> List[Dict]:
        """Parse a single PDF file."""
        logging.info(f"Parsing PDF: {pdf_path.name}")
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
        """Parse a single text file."""
        logging.info(f"Parsing TXT: {txt_path.name}")
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
        """Process parsed events for PDF source."""
        meet_name = pdf_path.stem.replace('-complete-results', '').replace('-', ' ').title()
        return self._process_events_common(events, meet_name, pdf_path.name, pdf_path.parent.name)
    
    def _process_events_for_txt(self, events: List[Dict], txt_path: Path) -> List[Dict]:
        """Process parsed events for text file source."""
        meet_name = txt_path.stem.replace('-complete-results', '').replace('-', ' ').title()
        return self._process_events_common(events, meet_name, txt_path.name, txt_path.parent.name)
    
    def _process_events_common(self, events: List[Dict], meet_name: str, source_file: str, meet_category: str) -> List[Dict]:
        """Common event processing logic for both PDF and txt sources."""
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
        """Parse both PDF and text files."""
        if pdf_paths is None:
            pdf_paths = list(self.raw_pdf_dir.rglob("*.pdf"))
        if txt_paths is None:
            txt_paths = list(self.raw_txt_dir.rglob("*.txt"))
        
        total_files = len(pdf_paths) + len(txt_paths)
        logging.info(f"Parsing {len(pdf_paths)} PDF files and {len(txt_paths)} TXT files ({total_files} total)")
        
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
        
        logging.info(f"Successfully parsed {successful_parses}/{total_files} files")
        logging.info(f"Total events extracted: {len(all_events)}")
        
        if not all_events:
            logging.warning("No events were successfully parsed!")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_events)
        return df
    
    
    def save_parsed_data(self, df: pd.DataFrame) -> Path:
        """Save processed data and create individual meet DataFrames."""
        if df.empty:
            logging.warning("No data exists")
            return None
        
        output_path = self.processed_dir / "parsed_events.csv"
        df.to_csv(output_path, index=False)
        
        # Store individual meets as DataFrames in memory
        self._create_individual_meet_dataframes(df)
        
        logging.info(f"Saved processed data to: {output_path}")
        logging.info(f"Created {len(self.individual_meet_dfs)} individual meet DataFrames")
        
        return output_path
    
    def _create_individual_meet_dataframes(self, df: pd.DataFrame):
        """Create and store individual meet DataFrames."""
        if df.empty or 'meet' not in df.columns:
            return
        
        # Clear existing DataFrames
        self.individual_meet_dfs.clear()
        
        # Create DataFrames for each meet
        for meet_name in df['meet'].unique():
            meet_df = df[df['meet'] == meet_name].copy()
            
            # Use a clean key for the dictionary
            safe_meet_name = meet_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            
            # Store in memory
            self.individual_meet_dfs[safe_meet_name] = meet_df
            
            logging.debug(f"Created DataFrame for meet: {meet_name} ({len(meet_df)} events)")
    
    
    def clean_existing_data(self) -> Tuple[Path, Path]:
        """Clean existing parsed data (no prediction)."""
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
            clean_df = self.clean_dataframe(original_df)
            
            # Save clean data
            clean_path = self.save_clean_data(clean_df)
            
            return parsed_data_path, clean_path
            
        except Exception as e:
            logging.error(f"Failed to clean existing data: {e}")
            return None, None


    def run_pipeline(self) -> Tuple[Path, Path]:
        """Run the complete pipeline on both PDF and TXT files (no prediction)."""
        logging.info("Starting Complete Meet Data Pipeline")
        
        logging.info("Parsing all files (PDF and TXT)")
        original_df = self.parse_all_files()
        
        logging.info("Saving processed data")
        original_path = self.save_parsed_data(original_df)

        logging.info("Cleaning saved data")
        clean_df = self.clean_dataframe(original_df)

        logging.info("Saving clean data")
        clean_path = self.save_clean_data(clean_df)

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
            parse_path = pipeline.parse_all_files()
            
            if parse_path:
                print(f"\nSucces: Generated {parse_path}")
            else:
                print("\nFailed to parse files.")
       
        elif command == "--clean":
            # Clean existing parsed data only
            logging.info("Running clean-only")
            parse_path, clean_path = pipeline.clean_existing_data()
            
            if clean_path:
                print(f"\nSuccess: Generated {clean_path}")
            else:
                print("\nFailed to clean existing data.")
            
    else:
        # Default behavior: parse only (skip cleaning for now)
        parse_path, clean_path = pipeline.run_pipeline()
            
        if parse_path and clean_path:
            print(f"\nSuccess: Generated {parse_path.name}, {clean_path.name}")
        elif parse_path:
            print(f"\nPartial success: Generated {parse_path.name} only")
        else:
            print("\nPipeline failed.")


if __name__ == "__main__":
    main()
