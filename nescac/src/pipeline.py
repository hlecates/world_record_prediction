import logging
import re
import pandas as pd
import pdfplumber
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import utils


class TextParser:
    
    def __init__(self):
        # Event header pattern (shared)
        self.event_re = re.compile(r'^Event\s+(\d+)\s+(Women|Men)\s+(\d+)\s+Yard\s+([A-Za-z ]+)(?:\s+Time\s+Trial)?$')
        self.diving_re = re.compile(r'^Event\s+(\d+)\s+(Women|Men)\s+([13])\s+mtr\s+Diving')
        
        # TXT FORMAT PATTERNS
        self.txt_individual_finals_re = re.compile(
            r'^\s*(\d+)\s+(.+?)\s+([A-Z]{2})\s+(.+?)\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)(?:\s+(\d+))?\s*$'
        )
        
        self.txt_individual_prelims_re = re.compile(
            r'^\s*(\d+)\s+(.+?)\s+([A-Z]{2})\s+(.+?)\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)\s*$'
        )
        
        self.txt_relay_re = re.compile(
            r"""^\s*
                (\d+)                                         # rank
                \s+
                ([A-Za-z\s\-]+?(?:College|University)(?:-[A-Z]{2})?)  # team
                \s+'([A-Z])'                                  # relay letter
                \s+
                ([\d:.NTXb#&]+)                               # first time (seed/prelim)
                \s+
                ([\d:.NTXb#&]+)                               # second time (finals)
                (?:\s+(\d+))?                                 # optional points
                \s*$
            """, re.VERBOSE
        )
        
        # PDF FORMAT PATTERNS
        self.pdf_individual_finals_re = re.compile(
            r'^\s*(\d+)\s+([A-Za-z\',.\s-]+?)\s+([A-Z]{2})\s+([A-Za-z\s\-]+(?:-[A-Z]{2})?)\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)\s*(\d+)?\s*$'
        )
        
        self.pdf_individual_prelims_re = re.compile(
            r'^\s*(\d+)\s+([A-Za-z\',.\s-]+?)\s+([A-Z]{2})\s+([A-Za-z\s\-]+(?:-[A-Z]{2})?)\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)\s*$'
        )
        
        # Legacy PDF individual pattern (keep for backwards compatibility)
        self.pdf_individual_legacy_re = re.compile(
            r'^(\*?\d+|---)\s+([A-Za-z\',.-]+,\s+[A-Za-z\',.-]+)\s+([A-Z]{2})\s+([A-Za-z ]+(?:[A-Za-z])+)\s+([\d:.NTXb]+)\s+([\d:.NTXb]+)(?:\s+([\d:.NTXb]+))?(?:\s+(\d+))?'
        )
        
        self.pdf_relay_re = re.compile(
            r'^(\*?\d+|---)\s+([A-Za-z ]+)\s+([A-Z])\s+([\d:.NTXb]+)\s+([\d:.NTXb]+)(?:\s+([\d:.NTXb]+))?(?:\s+(\d+))?'
        )

    def _get_patterns_for_format(self, source_format: str) -> dict:
        if source_format == 'txt':
            return {
                'individual_finals': self.txt_individual_finals_re,
                'individual_prelims': self.txt_individual_prelims_re,
                'relay': self.txt_relay_re
            }
        else:  # pdf
            return {
                'individual_finals': self.pdf_individual_finals_re,
                'individual_prelims': self.pdf_individual_prelims_re,
                'individual_legacy': self.pdf_individual_legacy_re,
                'relay': self.pdf_relay_re
            }
        

    def preprocess_text(self, text: str, source_format: str = 'auto') -> str:
        if source_format == 'auto':
            source_format = self._detect_format(text)
        
        if source_format == 'txt':
            return self._preprocess_txt_format(text)
        else:  # pdf or unknown
            return self._preprocess_pdf_format(text)
    
    def _detect_format(self, text: str) -> str:
        # Look for TXT-specific formatting patterns
        txt_indicators = [
            'Championship Final',
            'Consolation Final',
            'Preconsolation Final',
            'Preliminaries',
            '==============================================================================='
        ]
        
        if any(indicator in text for indicator in txt_indicators):
            return 'txt'
        
        # Look for PDF-specific artifacts
        pdf_indicators = [
            'HY-TEK\'S MEET MANAGER',
            'Licensed to',
            'COMPLETE RESULTS'
        ]
        
        if any(indicator in text for indicator in pdf_indicators):
            return 'pdf'
        
        return 'txt'  # Default to txt format for better parsing
    
    def _preprocess_txt_format(self, text: str) -> str:
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            # Keep original line structure but strip trailing whitespace
            line = line.rstrip()
            
            # Skip empty lines
            if not line.strip():
                continue
            
            # Skip common header lines that aren't useful
            skip_patterns = [
                r'^Licensed to',
                r'^HY-TEK\'S MEET MANAGER',
                r'^\d{4}.*Championships.*Results$',
                r'^={40,}$',  # Very long equal signs (table separators)
                r'^-{40,}$',  # Very long dashes
                r'^\s*Page\s+\d+',
                r'^\s*www\.',
                r'^\s*NESCAC:\s*\*',  # Record lines
                r'^\s*Pool:\s*#',     # Pool record lines
                r'^\s*Meet:\s*&',     # Meet record lines
                r'^\s*\d+\.\d+\s+NAT[AB]$',  # Qualifying times
            ]
            
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                continue
            
            processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def _preprocess_pdf_format(self, text: str) -> str:
        # For now, minimal preprocessing since PDF extraction seems fine
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.rstrip()
            if not line.strip():
                continue
            
            # Skip common PDF headers
            skip_patterns = [
                r'^Licensed to',
                r'^HY-TEK\'S MEET MANAGER',
                r'^COMPLETE RESULTS$',
                r'^\s*Results\s*$',
                r'^\d{4}.*Championship.*\d{4}$',
                r'^={20,}$',
                r'^-{20,}$'
            ]
            
            if any(re.match(pattern, line) for pattern in skip_patterns):
                continue
            
            processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def _is_section_header(self, line: str) -> Optional[str]:
        line_clean = line.strip().lower()

        # TXT format section headers
        finals_indicators = [
            'championship final',
            'consolation final', 
            'preconsolation final',
            'final',
            'championship',
            'consolation',
            'preconsolation',
        ]

        prelims_indicators = [
            'preliminaries',
            'prelim'
        ]

        # Recognize 'A - Final', 'B - Final', etc.
        if re.match(r'^[a-z] - final$', line_clean):
            return 'finals'
        if re.match(r'^[a-z] - championship$', line_clean):
            return 'finals'
        if re.match(r'^[a-z] - consolation$', line_clean):
            return 'finals'
        if re.match(r'^[a-z] - preconsolation$', line_clean):
            return 'finals'

        # Check TXT format first
        for indicator in finals_indicators:
            if indicator in line_clean:
                return 'finals'
        for indicator in prelims_indicators:
            if indicator in line_clean:
                return 'prelims'

        # PDF format (legacy support)
        if line_clean.startswith('finals') or 'final' in line_clean:
            return 'finals'
        elif line_clean.startswith('prelim'):
            return 'prelims'

        return None
    

    def _parse_relay_entry(self, line: str, source_format: str, is_exhibition: bool) -> Optional[Dict]:
        patterns = self._get_patterns_for_format(source_format)
        
        if source_format == 'txt':
            # Skip split time lines for TXT relays
            if re.match(r'^\s*(?:\d+(?:\.\d+)?\s+){3,}\d+(?:\.\d+)?\s*$', line):
                return None
            
            m = patterns['relay'].match(line)
            if m:
                return {
                    'raw': line,
                    'rank': m.group(1),
                    'team': m.group(2).strip(),
                    'relay_letter': m.group(3),
                    'entry_type': 'relay',
                    'seed_time': m.group(4),
                    'finals_time': m.group(5),
                    'points': m.group(6) if m.group(6) else None,
                    'prelim_time': None,
                    'exhibition': is_exhibition
                }
        
        else:  # PDF format
            m = patterns['relay'].match(line)
            if m:
                return {
                    'raw': line,
                    'rank': m.group(1),
                    'team': m.group(2).strip(),
                    'relay_letter': m.group(3),
                    'entry_type': 'relay',
                    'seed_time': m.group(4),
                    'finals_time': m.group(5),
                    'points': m.group(7) if len(m.groups()) >= 7 else None,
                    'prelim_time': None,
                    'exhibition': is_exhibition
                }
        
        return None

    def _parse_individual_entry(self, line: str, current_section: str, source_format: str, is_exhibition: bool) -> Optional[Dict]:
        patterns = self._get_patterns_for_format(source_format)
        
        # Clean the line for exhibition entries
        clean_line = line
        if is_exhibition:
            if source_format == 'txt':
                clean_line = re.sub(r'^\s*--\s+', '  ', line)
            else:  # PDF format
                # For PDF, don't clean the line since --- is part of the rank field
                clean_line = line
        
        if source_format == 'txt':
            # TXT format parsing (keep existing logic)
            if current_section == 'prelims':
                m = patterns['individual_prelims'].match(clean_line)
                if m:
                    rank, name, yr, school, seed_time, prelim_time = m.groups()
                    return {
                        'raw': line,
                        'rank': rank,
                        'name': name.strip(),
                        'yr': yr,
                        'school': school.strip(),
                        'entry_type': 'individual',
                        'exhibition': is_exhibition,
                        'seed_time': seed_time,
                        'prelim_time': prelim_time,
                        'finals_time': None,
                        'points': None
                    }
            else:  # finals section
                m = patterns['individual_finals'].match(clean_line)
                if m:
                    rank, name, yr, school, seed_time, finals_time, points = m.groups()
                    return {
                        'raw': line,
                        'rank': rank,
                        'name': name.strip(),
                        'yr': yr,
                        'school': school.strip(),
                        'entry_type': 'individual',
                        'exhibition': is_exhibition,
                        'seed_time': seed_time,
                        'finals_time': finals_time,
                        'points': points,
                        'prelim_time': None
                    }
        
        else:  # PDF format parsing
            # IMPORTANT: For PDF format, check if this is an exhibition entry
            # Exhibition entries should be skipped or handled separately
            if is_exhibition:
                return None  # Skip exhibition entries entirely for PDF
            
            # Try new PDF patterns first (only for non-exhibition entries)
            if current_section == 'prelims':
                m = patterns['individual_prelims'].match(clean_line)
                if m:
                    rank, name, yr, school, seed_time, prelim_time = m.groups()
                    return {
                        'raw': line,
                        'rank': rank,
                        'name': name.strip(),
                        'yr': yr,
                        'school': school.strip(),
                        'entry_type': 'individual',
                        'exhibition': False,  # Non-exhibition since we filtered them out
                        'seed_time': seed_time,
                        'prelim_time': prelim_time,
                        'finals_time': None,
                        'points': None
                    }
            else:  # finals section
                m = patterns['individual_finals'].match(clean_line)
                if m:
                    rank, name, yr, school, seed_time, finals_time, points = m.groups()
                    return {
                        'raw': line,
                        'rank': rank,
                        'name': name.strip(),
                        'yr': yr,
                        'school': school.strip(),
                        'entry_type': 'individual',
                        'exhibition': False,  # Non-exhibition since we filtered them out
                        'seed_time': seed_time,
                        'finals_time': finals_time,
                        'points': points,
                        'prelim_time': None
                    }
            
            # Fall back to legacy PDF pattern if new ones don't work 
            m = patterns['individual_legacy'].match(line)
            if m:
                # Check if the rank indicates exhibition
                rank = m.group(1)
                if rank == '---':
                    logging.debug(f"Skipping exhibition entry (legacy pattern): {line}")
                    return None
                
                entry = {
                    'raw': line,
                    'rank': rank,
                    'name': m.group(2).strip(),
                    'yr': m.group(3),
                    'school': m.group(4).strip(),
                    'entry_type': 'individual',
                    'exhibition': False
                }
                
                # Handle time assignments based on current section
                if current_section == 'finals':
                    entry['seed_time'] = None
                    entry['prelim_time'] = m.group(5)
                    entry['finals_time'] = m.group(6)
                    entry['points'] = m.group(8) if len(m.groups()) >= 8 else None
                else:
                    entry['seed_time'] = m.group(5)
                    entry['prelim_time'] = m.group(6)
                    entry['finals_time'] = m.group(7) if len(m.groups()) >= 7 else None
                    entry['points'] = m.group(8) if len(m.groups()) >= 8 else None
                
                return entry
        
        return None

    def _parse_entry(self, line: str, current_section: str, event_type: str, source_format: str) -> Optional[Dict]:
        if not line.strip():
            return None

        # Skip separator lines
        if re.match(r'^\s*-{5,}\s*$', line):
            return None
        
        # Skip split time lines (just numbers)
        if re.match(r'^\s*(?:\d+\.\d+\s+){2,}\s*$', line):
            return None
            
        # Skip reaction time lines and other technical info
        if 'r:+' in line or 'Declared false start' in line or line.strip().startswith('r:'):
            return None
        
        is_exhibition = False
        if source_format == 'txt':
            # TXT format: exhibition entries start with --
            is_exhibition = line.strip().startswith('--')
        else:  # PDF format
            # PDF format: exhibition entries start with --- (in rank position)
            is_exhibition = line.strip().startswith('---') or ' X' in line
        
        # Choose appropriate parser
        if event_type == 'relay':
            return self._parse_relay_entry(line, source_format, is_exhibition)
        else:
            return self._parse_individual_entry(line, current_section, source_format, is_exhibition)
    
    def parse_meet_text(self, text: str, source_format: str = 'auto') -> List[Dict]:
        # Detect format if auto
        if source_format == 'auto':
            source_format = self._detect_format(text)
        
        # Preprocess based on format
        processed_text = self.preprocess_text(text, source_format)
        
        # Use format-aware parsing logic
        return self._parse_processed_text(processed_text, source_format)
    
    def _parse_processed_text(self, text: str, source_format: str) -> List[Dict]:
        # Event tracking dictionary - key: (event_num, gender, distance, stroke)
        events_dict = {}
        
        # State variables
        current_event_key = None
        current_section = None
        
        def get_event_key(event_line: str) -> Optional[Tuple[str, str, int, str]]:
            match = self.event_re.match(event_line)
            if match:
                event_num, gender, distance, stroke = match.groups()
                # Skip Time Trial events and diving
                if ('time trial' in event_line.lower() or 
                    self.is_diving_event(event_line)):
                    return None
                return (event_num, gender, int(distance), stroke.strip())
            return None
        
        def is_any_skipped_event(event_line: str) -> bool:
            return (self.is_diving_event(event_line) or 
                    'time trial' in event_line.lower() or
                    'swim-off' in event_line.lower())
        
        def ensure_event_exists(event_key: Tuple, event_line: str):
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
        
        # Main parsing loop
        lines = text.splitlines()
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if this is ANY event header (including ones we want to skip)
            if line_stripped.startswith('Event'):
                # Check if we should skip this event
                if is_any_skipped_event(line_stripped):
                    current_event_key = None  # CRITICAL: Clear the current context
                    current_section = None
                    continue
                
                # Check if this is a valid swimming event
                event_key = get_event_key(line_stripped)
                if event_key:
                    current_event_key = event_key
                    ensure_event_exists(event_key, line_stripped)
                    current_section = None  # Reset section when new event found
                    continue
                else:
                    current_event_key = None
                    current_section = None
                    continue
            
            # Check for section header (only process if we have a valid current event)
            if current_event_key:
                section = self._is_section_header(line)
                if section:
                    current_section = section
                    continue
            
            # Try to parse entry (only if we have a valid current event and section)
            if current_event_key and current_section:
                event_type = events_dict[current_event_key]['event_type']
                entry = self._parse_entry(line, current_section, event_type, source_format)
                if entry:
                    # Handle relay vs individual events differently
                    if event_type == 'relay':
                        events_dict[current_event_key]['results'].append(entry)
                    else:
                        events_dict[current_event_key][current_section].append(entry)
                    continue
        
        # Convert events_dict to list format and return
        events = list(events_dict.values())
        
        # Log summary
        logging.debug(f"Parsed {len(events)} events from {source_format.upper()} format")
        
        # Print summary of parsed events
        total_events = len(events)
        individual_events = [e for e in events if e.get('event_type') == 'individual']
        relay_events = [e for e in events if e.get('event_type') == 'relay']
        
        total_finals = sum(len(e.get('finals', [])) for e in individual_events)
        total_prelims = sum(len(e.get('prelims', [])) for e in individual_events)
        total_relay_results = sum(len(e.get('results', [])) for e in relay_events)
        
        logging.debug(f"  Total unique events: {total_events}")
        logging.debug(f"  Individual events: {len(individual_events)} (Finals: {total_finals}, Prelims: {total_prelims})")
        logging.debug(f"  Relay events: {len(relay_events)} (Results: {total_relay_results})")
        
        for e in events:
            if e.get('event_type') == 'relay':
                logging.debug(f"RELAY Event: {e['event']} | Results: {len(e.get('results', []))}")
            else:
                logging.debug(f"INDIVIDUAL Event: {e['event']} | Finals: {len(e.get('finals', []))} | Prelims: {len(e.get('prelims', []))}")
        
        return events
    
    def is_diving_event(self, event_line: str) -> bool:
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
        return self._process_events_common(events, meet_name, pdf_path.name, pdf_path.parent.name)
    
    def _process_events_for_txt(self, events: List[Dict], txt_path: Path) -> List[Dict]:
        meet_name = txt_path.stem.replace('-complete-results', '').replace('-', ' ').title()
        return self._process_events_common(events, meet_name, txt_path.name, txt_path.parent.name)
    
    def _process_events_common(self, events: List[Dict], meet_name: str, source_file: str, meet_category: str) -> List[Dict]:
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
    
    
    def save_parsed_data(self, df: pd.DataFrame) -> Path:
        if df.empty:
            logging.warning("No data exists")
            return None
        
        output_path = self.processed_dir / "parsed_events.csv"
        df.to_csv(output_path, index=False)
        
        # Store individual meets as DataFrames in memory
        self._create_individual_meet_dataframes(df)
        
        logging.debug(f"Saved processed data to: {output_path}")
        
        return output_path
    
    def _create_individual_meet_dataframes(self, df: pd.DataFrame):
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
            df = pipeline.parse_all_files()  # This returns a DataFrame
            
            if not df.empty:  # Check if DataFrame has data
                # Save the parsed data
                parse_path = pipeline.save_parsed_data(df)
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
