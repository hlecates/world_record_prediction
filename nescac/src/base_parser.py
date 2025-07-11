import logging
import re
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod


class BaseParser(ABC):
    """Base class for parsing swimming meet results from different file formats."""
    
    def __init__(self):
        # Event header pattern (shared across formats)
        self.event_re = re.compile(r'^Event\s+(\d+)\s+(Women|Men)\s+(\d+)\s+Yard\s+([A-Za-z ]+)(?:\s+Time\s+Trial)?$')
        self.diving_re = re.compile(r'^Event\s+(\d+)\s+(Women|Men)\s+([13])\s+mtr\s+Diving')
    
    def is_diving_event(self, event_line: str) -> bool:
        """Check if an event line represents a diving event."""
        return self.diving_re.match(event_line) is not None
    
    def _cleanup_school_time(self, entry: Dict) -> Dict:
        """Clean up school names that have embedded times."""
        school = entry.get('school')
        if school:
            time_match = re.search(
                r'\s+([\d]+:[\d]+\.[\d]+|[\d]+\.[\d]+)\s*$',
                school
            )
            if time_match:
                # remove the embedded time from school
                entry['school'] = school[:time_match.start()].strip()
                # always overwrite finals_time
                entry['finals_time'] = time_match.group(1)
        return entry
    
    def _get_event_key(self, event_line: str) -> Optional[Tuple[str, str, int, str]]:
        """Extract event key from event header line."""
        match = self.event_re.match(event_line)
        if match:
            event_num, gender, distance, stroke = match.groups()
            # Skip Time Trial events and diving
            if ('time trial' in event_line.lower() or 
                self.is_diving_event(event_line)):
                return None
            logging.debug(f">>> Found event: {event_line}")
            return (event_num, gender, int(distance), stroke.strip())
        return None
    
    def _is_any_skipped_event(self, event_line: str) -> bool:
        """Check if event should be skipped (diving, time trial, etc.)."""
        return (self.is_diving_event(event_line) or 
                'time trial' in event_line.lower() or
                'swim-off' in event_line.lower())
    
    def _ensure_event_exists(self,
                             events_dict: Dict,
                             event_key: Tuple,
                             event_line: str):
        """Initialize storage for a new event."""
        if event_key in events_dict:
            return

        is_relay = any(w in event_line.lower()
                       for w in ('relay', 'medley relay', 'freestyle relay'))

        if is_relay:
            # Relays will just accumulate into a `results` list
            events_dict[event_key] = {
                'event': event_line,
                'results': [],
                'event_type': 'relay'
            }
        else:
            # Individuals get a results_map to merge prelims+finals
            events_dict[event_key] = {
                'event': event_line,
                'results_map': {},
                'event_type': 'individual'
            }

    def _consolidate_swimmer_results(self, events_dict: Dict) -> Dict:
        """
        For each individual event, convert its results_map into a flat results list.
        """
        for key, data in events_dict.items():
            if data.get('event_type') == 'individual':
                # Move the merged map into a list
                merged = list(data.pop('results_map', {}).values())
                data['results'] = merged
        return events_dict

    @abstractmethod
    def parse_meet_text(self, text: str) -> List[Dict]:
        """Parse meet text and return list of events with results."""
        pass
    
    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        """Preprocess raw text before parsing."""
        pass
    
    @abstractmethod
    def _is_section_header(self, line: str) -> Optional[str]:
        """Determine if line is a section header and return section type."""
        pass
    
    @abstractmethod
    def _parse_entry(self, line: str, current_section: str, event_type: str, **kwargs) -> Optional[Dict]:
        """Parse a single data entry line."""
        pass