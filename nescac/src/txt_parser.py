import logging
import re
from typing import List, Dict, Optional, Tuple
from base_parser import BaseParser


class TXTParser(BaseParser):
    """Parser for TXT format swimming meet results with unified results list."""
    def __init__(self):
        super().__init__()
        # Pattern for prelims entries (rank, name, year, school, seed, prelim) - updated for robust year parsing
        # Year field is now optional and can handle missing years, two-digit years, etc.
        # Improved to better handle field boundaries when year is missing
        self.txt_prelims_re = re.compile(
            r'^\s*(\d+)\s+([A-Za-z\s,]+?)\s+([A-Za-z0-9]{1,4})\s+([A-Za-z\s]+?)\s+([\d:.NTXb#&]+)(?:\s+([\d:.NTXb#&]+))?\s*$'
        )
        # Pattern for finals entries (rank, name, year, school, prelim, final, points) - updated for robust year parsing
        self.txt_finals_re = re.compile(
            r'^\s*(\d+)\s+([A-Za-z\s,]+?)\s+([A-Za-z0-9]{1,4})\s+([A-Za-z\s]+?)\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)(?:\s+(\d+))?\s*$'
        )
        # Pattern for entries with only one time field - updated for robust year parsing
        self.txt_single_time_re = re.compile(
            r'^\s*(\d+)\s+([A-Za-z\s,]+?)\s+([A-Za-z0-9]{1,4})\s+([A-Za-z\s]+?)\s+([\d:.NTXb#&]+)\s*$'
        )
        # Fallback patterns for lines with NO year field - these are more specific to handle missing years
        # These patterns look for specific spacing patterns that indicate no year field
        self.txt_prelims_no_year_re = re.compile(
            r'^\s*(\d+)\s+([A-Za-z\s,]+?)\s{2,}([A-Za-z\s]+?)\s+([\d:.NTXb#&]+)(?:\s+([\d:.NTXb#&]+))?\s*$'
        )
        self.txt_finals_no_year_re = re.compile(
            r'^\s*(\d+)\s+([A-Za-z\s,]+?)\s{2,}([A-Za-z\s]+?)\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)(?:\s+(\d+))?\s*$'
        )
        self.txt_single_time_no_year_re = re.compile(
            r'^\s*(\d+)\s+([A-Za-z\s,]+?)\s{2,}([A-Za-z\s]+?)\s+([\d:.NTXb#&]+)\s*$'
        )

    def preprocess_text(self, text: str) -> str:
        """Remove noise lines and strip trailing whitespace."""
        lines = text.split("\n")
        skip_patterns = [
            r"^Licensed to",
            r"^HY-TEK'S MEET MANAGER",
            r"^\d{4}.*Championships.*Results$",
            r"^={40,}$",
            r"^\s*Page\s+\d+",
            r"^\s*www\.",
            r"^\s*NESCAC:\s*\*",
            r"^\s*Pool:\s*[#@]",
            r"^\s*Meet:\s*&",
            r"^\s*\d+\.\d+\s+NAT[AB]$",
            r"^\s*Name\s+Year\s+School",
            r"^\s*Name\s+Year\s+School\s+Prelims\s+Finals\s+Points",
            r"^\s*Name\s+Year\s+School\s+Seed\s+Prelims",
        ]
        processed = []
        for line in lines:
            line = line.rstrip()
            if not line.strip():
                continue
            if any(re.match(p, line, re.IGNORECASE) for p in skip_patterns):
                continue
            processed.append(line)
        return "\n".join(processed)

    def _is_section_header(self, line: str) -> Optional[str]:
        """Identify prelims/finals section headers."""
        lc = line.strip().lower()
        exact = {
            'championship final': 'finals',
            'consolation final': 'finals',
            'preconsolation final': 'finals',
            'preliminaries': 'prelims',
            'final': 'finals',
            'finals': 'finals',
            'prelim': 'prelims',
            'prelims': 'prelims'
        }
        if lc in exact:
            return exact[lc]
        if 'preliminaries' in lc:
            return 'prelims'
        if any(ind in lc for ind in ['championship final', 'consolation final', 'preconsolation final']):
            return 'finals'
        if re.match(r'^[a-z] - final$', lc) or re.match(r'^[a-z] - consolation$', lc):
            return 'finals'
        return None

    def _clean_time_string(self, time_str: str) -> str:
        """Remove NATA/NATB indicators and # from time strings."""
        if not time_str:
            return time_str
        # Remove NATA/NATB indicators and #
        cleaned = re.sub(r'\s*NAT[AB]\s*$', '', time_str.strip())
        cleaned = cleaned.replace('#', '')
        return cleaned.strip()
    
    def _cleanup_school_time(self, entry: Dict) -> Dict:
        """Extract stray time from school field if present and use it as the actual time."""
        school = entry.get('school') or ''
        m = re.search(r"\s+([\d]+:[\d]+\.[\d]+|[\d]+\.[\d]+)\s*$", school)
        if m:
            extracted_time = m.group(1)
            entry['school'] = school[:m.start()].strip()
            
            # Use the extracted time as the appropriate time field
            if entry.get('finals_time') and not entry.get('prelim_time'):
                # If we have finals_time but no prelim_time, the extracted time is likely prelim
                entry['prelim_time'] = extracted_time
            elif entry.get('prelim_time') and not entry.get('finals_time'):
                # If we have prelim_time but no finals_time, the extracted time is likely finals
                entry['finals_time'] = extracted_time
            elif not entry.get('prelim_time') and not entry.get('finals_time'):
                # If we have no times at all, use the extracted time as prelim
                entry['prelim_time'] = extracted_time
            else:
                # If we have both times, the extracted time might be a better finals time
                # Check if the extracted time is faster (better) than current finals time
                try:
                    extracted_float = float(extracted_time.replace(':', ''))
                    current_finals = entry.get('finals_time', '')
                    if current_finals:
                        current_float = float(current_finals.replace(':', ''))
                        if extracted_float < current_float:
                            entry['finals_time'] = extracted_time
                except (ValueError, TypeError):
                    # If we can't compare, just use as prelim if no prelim exists
                    if not entry.get('prelim_time'):
                        entry['prelim_time'] = extracted_time
        return entry

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
        
        # If it's exactly 2 characters and both are digits, it's likely a year code (e.g., '05', '06')
        if len(year_str) == 2 and year_str.isdigit():
            return year_str
        
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

    def _parse_individual_entry(self, line: str, current_section: str, is_exhibition: bool) -> Optional[Dict]:
        """Parse a single individual entry line."""
        clean = line.strip()
        # Try finals pattern first (with year)
        m = self.txt_finals_re.match(clean)
        if m:
            groups = m.groups()
            rank, name, yr, school, prelim, final, pts = groups
            parsed_yr = self._parse_year_field(yr) if yr else 'NONE'
            result = {
                'name': name.strip(),
                'yr': parsed_yr,
                'school': school.strip(),
                'seed_time': None,
                'prelim_time': self._clean_time_string(prelim) if prelim else None,
                'finals_time': self._clean_time_string(final) if final else None,
                'rank': 'exhibition' if is_exhibition else rank,
                'exhibition': is_exhibition,
                'section': current_section
            }
            result = self._cleanup_school_time(result)
            return result
        # Try finals pattern (no year)
        m = self.txt_finals_no_year_re.match(clean)
        if m:
            rank, name, school, prelim, final, pts = m.groups()
            result = {
                'name': name.strip(),
                'yr': 'NONE',
                'school': school.strip(),
                'seed_time': None,
                'prelim_time': self._clean_time_string(prelim) if prelim else None,
                'finals_time': self._clean_time_string(final) if final else None,
                'rank': 'exhibition' if is_exhibition else rank,
                'exhibition': is_exhibition,
                'section': current_section
            }
            result = self._cleanup_school_time(result)
            return result
        # Try prelims pattern (with year)
        m = self.txt_prelims_re.match(clean)
        if m:
            groups = m.groups()
            rank, name, yr, school, time1, time2 = groups
            parsed_yr = self._parse_year_field(yr) if yr else 'NONE'
            if time2:
                seed_time = self._clean_time_string(time1)
                prelim_time = self._clean_time_string(time2)
            else:
                seed_time = None
                prelim_time = self._clean_time_string(time1)
            result = {
                'name': name.strip(),
                'yr': parsed_yr,
                'school': school.strip(),
                'seed_time': seed_time,
                'prelim_time': prelim_time,
                'finals_time': None,
                'rank': 'exhibition' if is_exhibition else rank,
                'exhibition': is_exhibition,
                'section': current_section
            }
            result = self._cleanup_school_time(result)
            return result
        # Try prelims pattern (no year)
        m = self.txt_prelims_no_year_re.match(clean)
        if m:
            rank, name, school, time1, time2 = m.groups()
            if time2:
                seed_time = self._clean_time_string(time1)
                prelim_time = self._clean_time_string(time2)
            else:
                seed_time = None
                prelim_time = self._clean_time_string(time1)
            result = {
                'name': name.strip(),
                'yr': 'NONE',
                'school': school.strip(),
                'seed_time': seed_time,
                'prelim_time': prelim_time,
                'finals_time': None,
                'rank': 'exhibition' if is_exhibition else rank,
                'exhibition': is_exhibition,
                'section': current_section
            }
            result = self._cleanup_school_time(result)
            return result
        # Try single time pattern (with year)
        m = self.txt_single_time_re.match(clean)
        if m:
            groups = m.groups()
            rank, name, yr, school, time = groups
            parsed_yr = self._parse_year_field(yr) if yr else 'NONE'
            if current_section == 'prelims':
                prelim_time = self._clean_time_string(time)
                seed_time = None
                finals_time = None
            else:
                prelim_time = None
                seed_time = None
                finals_time = self._clean_time_string(time)
            result = {
                'name': name.strip(),
                'yr': parsed_yr,
                'school': school.strip(),
                'seed_time': seed_time,
                'prelim_time': prelim_time,
                'finals_time': finals_time,
                'rank': 'exhibition' if is_exhibition else rank,
                'exhibition': is_exhibition,
                'section': current_section
            }
            result = self._cleanup_school_time(result)
            return result
        # Try single time pattern (no year)
        m = self.txt_single_time_no_year_re.match(clean)
        if m:
            rank, name, school, time = m.groups()
            if current_section == 'prelims':
                prelim_time = self._clean_time_string(time)
                seed_time = None
                finals_time = None
            else:
                prelim_time = None
                seed_time = None
                finals_time = self._clean_time_string(time)
            result = {
                'name': name.strip(),
                'yr': 'NONE',
                'school': school.strip(),
                'seed_time': seed_time,
                'prelim_time': prelim_time,
                'finals_time': finals_time,
                'rank': 'exhibition' if is_exhibition else rank,
                'exhibition': is_exhibition,
                'section': current_section
            }
            result = self._cleanup_school_time(result)
            return result
        return None

    def _parse_entry(self, line: str, current_section: str, event_type: str, **kwargs) -> Optional[Dict]:
        """Parse a single data entry line."""
        if not line.strip():
            return None
        # Skip separators and split-time lines
        if re.match(r'^\s*-{5,}\s*$', line) or re.match(r'^\s*(?:\d+\.\d+\s+){2,}\s*$', line):
            return None
        is_exhibition = line.strip().startswith('--')
        if event_type == 'relay':
            return None  # skip relays entirely
        return self._parse_individual_entry(line, current_section, is_exhibition)

    def parse_meet_text(self, text: str) -> List[Dict]:
        processed = self.preprocess_text(text)
        events: Dict[Tuple[str, str, int, str], Dict] = {}
        current_key = None
        current_section = None

        for line in processed.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('Event'):
                if self._is_any_skipped_event(stripped):
                    current_key = None; current_section = None
                    continue
                key = self._get_event_key(stripped)
                if key:
                    if key != current_key:
                        current_key = key
                        self._ensure_event_exists(events, key, stripped)
                        logging.debug(f"TXT: Starting event {key} - {stripped}")
                    current_section = None
                    continue
                current_key = None; current_section = None
                continue
            if current_key:
                sec = self._is_section_header(line)
                if sec:
                    current_section = sec
                    logging.debug(f"TXT: Section change to '{sec}' for event {current_key}")
                    continue
            if current_section and current_key:
                entry = self._parse_entry(line, current_section, events[current_key]['event_type'])
                if not entry:
                    continue
                # merge into unified map
                result_map = events[current_key].setdefault('results_map', {})
                key_tuple = (entry['name'], entry['yr'], entry['school'])
                existing = result_map.get(key_tuple)
                if not existing:
                    existing = {
                        'name': entry['name'],
                        'yr': entry['yr'],
                        'school': entry['school'],
                        'exhibition': entry['exhibition'],
                        'seed_time': None,
                        'prelim_time': None,
                        'finals_time': None,
                        'prelim_rank': None,
                        'final_rank': None
                    }
                    result_map[key_tuple] = existing
                else:
                    if existing['school'] != entry['school']:
                        logging.warning(f"School mismatch for {entry['name']} ({entry['yr']}): {existing['school']} vs {entry['school']}")
                
                # Update based on section
                if current_section == 'prelims':
                    existing['seed_time'] = existing['seed_time'] or entry['seed_time']
                    existing['prelim_time'] = existing['prelim_time'] or entry['prelim_time']
                    existing['prelim_rank'] = existing['prelim_rank'] or entry['rank']
                    logging.debug(f"TXT: Prelims entry for {entry['name']}: seed={entry['seed_time']}, prelim={entry['prelim_time']}, rank={entry['rank']}")
                else:  # finals
                    # In finals, the first time is prelim, second is final
                    if entry['prelim_time'] and not existing['prelim_time']:
                        existing['prelim_time'] = entry['prelim_time']
                    existing['finals_time'] = entry['finals_time']
                    existing['final_rank'] = entry['rank']
                    logging.debug(f"TXT: Finals entry for {entry['name']}: prelim={entry['prelim_time']}, finals={entry['finals_time']}, rank={entry['rank']}")

        # consolidate and return
        self._consolidate_swimmer_results(events)
        return list(events.values())
