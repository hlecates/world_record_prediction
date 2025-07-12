import logging
import re
from typing import List, Dict, Optional, Tuple
from base_parser import BaseParser


class PDFParser(BaseParser):
    """Parser for PDF-extracted swimming meet results with unified results list."""
    def __init__(self):
        super().__init__()
        # PDF-specific regex patterns for individual events - updated for robust year parsing
        self.pdf_individual_re = re.compile(
            r'^\s*(\*?\d+)\s+([A-Za-z\',.\s\-]+?)\s+([A-Za-z0-9]{1,4})\s+([A-Za-z\s\-]+(?:-[A-Z]{2})?)\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)(?:\s+(\d+))?\s*$'
        )
        # More flexible regex for lines with qualifying indicators
        self.pdf_individual_flexible_re = re.compile(
            r'^\s*(\*?\d+)\s+([A-Za-z\',.\s\-]+?)\s+([A-Za-z0-9]{1,4})\s+([A-Za-z\s\-]+(?:-[A-Z]{2})?)\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)(?:\s+[A-Za-z]+)?\s*$'
        )
        # PDF relay patterns
        self.pdf_relay_finals_re = re.compile(
            r'^\s*(\d+)\s+([A-Za-z\s\-]+)\s+([A-Z])\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)(?:\s+(\d+))?\s*$'
        )
        self.pdf_relay_re = re.compile(
            r'^(?:\*?\d+|---)\s+([A-Za-z ]+)\s+([A-Z])\s+([\d:.NTXb]+)\s+([\d:.NTXb]+)(?:\s+([\d:.NTXb]+))?(?:\s+(\d+))?$'
        )

    def preprocess_text(self, text: str) -> str:
        lines = text.split('\n')
        skip_patterns = [
            r"^Licensed to",
            r"^HY-TEK'S MEET MANAGER",
            r"^COMPLETE RESULTS$",
            r"^\s*Results\s*$",
            r"^\d{4}.*Championship.*\d{4}$",
            r"^={20,}$",
            r"^-{20,}$",
        ]
        processed = []
        for line in lines:
            line = line.rstrip()
            if not line.strip():
                continue
            if any(re.match(p, line) for p in skip_patterns):
                continue
            processed.append(line)
        return "\n".join(processed)

    def _is_section_header(self, line: str) -> Optional[str]:
        lc = line.strip().lower()
        if lc == 'finals':
            return 'finals'
        if lc == 'finals:':
            return 'finals'
        if lc.startswith('prelim'):
            return 'prelims'
        if lc == 'preliminaries':
            return 'prelims'
        if lc.startswith('bonus'):
            return 'finals'  # Bonus sections are typically finals
        # Ignore column headers that contain section keywords
        if 'name' in lc and ('yr' in lc or 'school' in lc):
            return None
        return None

    def _parse_year_field(self, year_str: str) -> str:
        """
        Parse year field robustly. Returns 'NONE' if no valid year found.
        
        Args:
            year_str: The year field from the regex match
            
        Returns:
            String representation of year or 'NONE' if invalid
        """
        if not year_str or not year_str.strip():
            return 'NONE'
        
        year_str = year_str.strip()
        
        # If it's longer than 4 characters, it's likely not a year
        if len(year_str) > 4:
            return 'NONE'
        
        # If it's exactly 2 characters and both are uppercase letters, it's likely a year
        if len(year_str) == 2 and year_str.isalpha() and year_str.isupper():
            return year_str
        
        # If it's exactly 2 characters and both are digits, it's likely NOT a year
        # In swimming results, 2-digit numbers like "04", "05", "06" are typically
        # missing year data, not actual years
        if len(year_str) == 2 and year_str.isdigit():
            return 'NONE'
        
        # If it's 3-4 characters and contains numbers, it's likely not a year
        if len(year_str) >= 3 and any(c.isdigit() for c in year_str):
            return 'NONE'
        
        # If it's 3-4 characters and all uppercase letters, it might be a year
        if len(year_str) >= 3 and year_str.isalpha() and year_str.isupper():
            return year_str
        
        # If it's 1 character and is a digit, it's likely not a year
        if len(year_str) == 1 and year_str.isdigit():
            return 'NONE'
        
        # If it's 1-2 characters and contains numbers but is longer than 2, it's likely not a year
        if len(year_str) > 2 and any(c.isdigit() for c in year_str):
            return 'NONE'
        
        # Default case: assume it's a year if it's 1-4 characters
        return year_str

    def _parse_individual_entry(
        self, line: str, section: str, is_exhibition: bool
    ) -> Optional[Dict]:
        clean = line.strip()
        
        # Try the strict regex first
        m = self.pdf_individual_re.match(clean)
        if m:
            rank, name, yr, school, seed, finals, pts = m.groups()
            # Parse year field robustly
            parsed_yr = self._parse_year_field(yr)
            logging.debug(f"PDF: Regex matched - rank='{rank}', name='{name}', yr='{yr}' -> parsed='{parsed_yr}', school='{school}', seed='{seed}', finals='{finals}', pts='{pts}'")
            
            # Determine if this is a finals entry based on points column
            is_finals_entry = pts is not None and pts.isdigit()
            
            # Handle the case where there's no points column
            if pts is None:
                pts = None
                finals = finals if finals else None
            else:
                # If we have points, the finals time is the second time column
                finals = finals if finals else None
            
            result = {
                'name': name.strip(),
                'yr': parsed_yr,
                'school': school.strip(),
                'seed_time': seed,
                'prelim_time': None,
                'finals_time': finals if is_finals_entry else None,
                'rank': rank,
                'exhibition': is_exhibition,
                'is_finals_entry': is_finals_entry
            }
            logging.debug(f"PDF: Parsed entry: {result}")
            return result
        
        # Try the flexible regex if the strict one failed
        m = self.pdf_individual_flexible_re.match(clean)
        if m:
            rank, name, yr, school, seed, finals = m.groups()
            # Parse year field robustly
            parsed_yr = self._parse_year_field(yr)
            logging.debug(f"PDF: Flexible regex matched - rank='{rank}', name='{name}', yr='{yr}' -> parsed='{parsed_yr}', school='{school}', seed='{seed}', finals='{finals}'")
            
            result = {
                'name': name.strip(),
                'yr': parsed_yr,
                'school': school.strip(),
                'seed_time': seed,
                'prelim_time': None,
                'finals_time': finals,
                'rank': rank,
                'exhibition': is_exhibition,
                'is_finals_entry': False
            }
            logging.debug(f"PDF: Parsed entry (flexible): {result}")
            return result
        
        logging.debug(f"PDF: Both regexes failed for line: '{clean}' in section '{section}'")
        return None

    def _parse_entry(self, line: str, current_section: str, event_type: str, **kwargs) -> Optional[Dict]:
        if not line.strip():
            return None
        # Skip separators and split-time lines
        if re.match(r'^\s*-{5,}\s*$', line) or re.match(r'^\s*(?:\d+\.\d+\s+){2,}\s*$', line):
            return None
        is_exhibition = line.strip().startswith('---')
        if event_type == 'relay':
            return None  # skip relays entirely
        return self._parse_individual_entry(line, current_section, is_exhibition)

    def parse_meet_text(self, text: str) -> List[Dict]:
        processed = self.preprocess_text(text)
        events: Dict[Tuple, Dict] = {}
        current_key = None
        current_section = None
        in_prelims = False

        # First pass: collect all raw entries for each event
        for line in processed.split("\n"):
            stripped = line.strip()
            if not stripped and not in_prelims:
                continue
            if stripped.startswith('Event'):
                if self._is_any_skipped_event(stripped):
                    current_key = None; current_section = None; in_prelims = False
                    continue
                key = self._get_event_key(stripped)
                if key:
                    if key != current_key:
                        current_key = key
                        self._ensure_event_exists(events, key, stripped)
                        logging.debug(f"PDF: Starting event {key} - {stripped}")
                        current_section = None; in_prelims = False
                    else:
                        logging.debug(f"PDF: Continuing event {key} - {stripped}")
                        current_section = None; in_prelims = False
                    continue
                current_key = None; current_section = None; in_prelims = False
                continue
            if current_key:
                sec = self._is_section_header(line)
                if sec:
                    current_section = sec
                    in_prelims = (sec == 'prelims')
                    logging.debug(f"PDF: Section change to '{sec}' for event {current_key}")
                    continue
            if current_key:
                entry = self._parse_entry(line, current_section or '', events[current_key]['event_type'])
                if not entry:
                    continue
                # Store all raw entries for the event
                raw_entries = events[current_key].setdefault('raw_entries', [])
                raw_entries.append(entry)

        # Second pass: consolidate entries for each event
        for key, data in events.items():
            if data.get('event_type') != 'individual':
                continue
            raw_entries = data.get('raw_entries', [])
            results_map = {}
            for entry in raw_entries:
                key_tuple = (entry['name'], entry['yr'], entry['school'])
                existing = results_map.get(key_tuple)
                if not existing:
                    existing = {
                        'name': entry['name'],
                        'yr': entry['yr'],
                        'school': entry['school'],
                        'exhibition': entry['exhibition'],
                        'seed_time': entry.get('seed_time'),
                        'prelim_time': None,
                        'finals_time': None,
                        'prelim_rank': None,
                        'final_rank': None
                    }
                    results_map[key_tuple] = existing
                # Assign finals_time if entry has a points column (i.e., is a finals entry)
                if entry.get('is_finals_entry', False):
                    # Heuristic: if entry has a points column, treat as finals
                    if not existing['finals_time']:
                        existing['finals_time'] = entry['finals_time']
                        existing['final_rank'] = entry['rank']
                        logging.debug(f"PDF: Finals entry (2nd pass) for {entry['name']}: finals={entry['finals_time']}, rank={entry['rank']}")
                else:
                    # Otherwise, treat as prelims
                    if not existing['prelim_time']:
                        existing['prelim_time'] = entry.get('finals_time') or entry.get('prelim_time')
                        existing['prelim_rank'] = entry['rank']
                        logging.debug(f"PDF: Prelims entry (2nd pass) for {entry['name']}: prelim={existing['prelim_time']}, rank={entry['rank']}")
            data['results_map'] = results_map

        # Debug output
        for key, data in events.items():
            if data.get('event_type') == 'individual':
                results_map = data.get('results_map', {})
                finals_count = sum(1 for entry in results_map.values() if entry.get('finals_time'))
                logging.debug(f"PDF: Event {key}: {finals_count}/{len(results_map)} entries have finals times")
        self._consolidate_swimmer_results(events)
        return list(events.values())
