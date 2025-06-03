import os
import glob
import pdfplumber
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SwimEvent:
    """Structure for swimming event data"""
    pdf_path: str
    meet_name: str
    meet_date: Optional[datetime]
    meet_location: Optional[str]
    event_number: int
    event_name: str
    gender: str
    distance: int
    stroke: str
    records: Dict[str, Dict[str, str]]
    entries: List[Dict]
    finals: List[Dict]

def get_processed_events(output_dir: Path) -> Set[Tuple[str, int, str]]:
    """Get set of already processed events to avoid duplicates."""
    processed = set()
    csv_path = output_dir / "compiled_events.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        processed.update(
            (row['pdf_path'], row['event_number'], row['stroke'])
            for _, row in df.iterrows()
        )
    return processed

def clean_time(time_str: str) -> Optional[str]:
    """Standardize time format to MM:SS.ss"""
    if not time_str:
        return None
    
    # Handle different time formats including minutes
    time_match = re.search(r'(?:(\d+):)?(\d+\.\d+)', time_str)
    if time_match:
        minutes = time_match.group(1) or "0"
        seconds = time_match.group(2)
        return f"{int(minutes):02d}:{float(seconds):.2f}"
    return None

def extract_meet_info(text: str) -> Dict:
    """Extract meet name, date and location."""
    info = {}
    
    # Look for meet name
    for line in text.splitlines():
        if any(x in line.upper() for x in ['CHAMPIONSHIPS', 'TRIALS', 'MEET']):
            info['name'] = line.strip()
            break
    
    # Look for date
    date_match = re.search(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:-\d{1,2})?,\s+\d{4}', text)
    if date_match:
        try:
            info['date'] = datetime.strptime(date_match.group(0), '%B %d, %Y')
        except ValueError:
            pass
    
    # Look for location
    loc_match = re.search(r'(?:at|in)\s+([^,]+),\s*([A-Z]{2})', text)
    if loc_match:
        info['location'] = f"{loc_match.group(1)}, {loc_match.group(2)}"
    
    return info

def extract_records(section: str) -> Dict[str, Dict[str, str]]:
    """Extract swimming records with standardized format."""
    records = {}
    record_patterns = {
        'world': r'World:\s*(\d+:\d+\.\d+)\s*(?:W)?\s+(\d+/\d+/\d+)\s+([A-Za-z\s]+)',
        'american': r'American:\s*(\d+:\d+\.\d+)\s*(?:A)?\s+(\d+/\d+/\d+)\s+([A-Za-z\s]+)',
        'us_open': r'U\.S\. Open:\s*(\d+:\d+\.\d+)\s*(?:O)?\s+(\d+/\d+/\d+)\s+([A-Za-z\s]+)'
    }
    
    for record_type, pattern in record_patterns.items():
        if match := re.search(pattern, section, re.IGNORECASE):
            records[record_type] = {
                'time': clean_time(match.group(1)),
                'date': match.group(2),
                'holder': match.group(3).strip()
            }
    
    return records

def parse_entries(section: str) -> List[Dict]:
    """Parse entries from swim meet results."""
    entries = []
    
    # Modified pattern to handle single-line entries
    entry_pattern = (
        r'([A-Za-z\s\'-]+?)\s+'  # Name (non-greedy match)
        r'(\d+:\d+\.\d+)\s*'     # Final time
        r'(\d+)\s*'              # Age
        r'([A-Za-z-]+(?:-[A-Z]{2})?)\s*'  # Team (with optional region code)
        r'(\d+:\d+\.\d+)'        # Seed/Entry time
    )
    
    print("\nParsing entries:")
    for line in section.split('\n'):
        line = line.strip()
        
        # Skip split times and reaction times
        if line.startswith('r:+') or '(' in line:
            continue
            
        if match := re.search(entry_pattern, line):
            name, final_time, age, team, entry_time = match.groups()
            print(f"Found entry: {name.strip()} - {final_time}")
            
            result = {
                'name': name.strip(),
                'age': int(age),
                'team': team.strip(),
                'entry_time': clean_time(entry_time),
                'final_time': clean_time(final_time)
            }
            entries.append(result)
    
    return entries

def extract_event_info(pdf_path: str, processed_events: Set[Tuple]) -> List[SwimEvent]:
    """Extract structured event information from PDF."""
    events = []
    
    with pdfplumber.open(pdf_path) as pdf:
        text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
        
        # Get meet information
        meet_info = extract_meet_info(text)
        
        # Split into event sections
        sections = re.split(r'EVENT\s*:\s*\d+', text)
        
        for section in sections[1:]:  # Skip header
            try:
                # Parse event details
                event_match = re.search(
                    r'(Men|Women).*?(\d+)\s*(?:LC|Meter|Meters)?\s*([^(]+)',
                    section
                )
                if not event_match:
                    continue
                
                gender, distance, stroke = event_match.groups()
                event_num = len(events) + 1
                
                # Check if already processed
                event_key = (pdf_path, event_num, stroke.strip())
                if event_key in processed_events:
                    print(f"Skipping already processed: Event {event_num} {stroke.strip()}")
                    continue
                
                # Create event object
                event = SwimEvent(
                    pdf_path=pdf_path,
                    meet_name=meet_info.get('name', ''),
                    meet_date=meet_info.get('date'),
                    meet_location=meet_info.get('location'),
                    event_number=event_num,
                    event_name=f"{distance} {stroke.strip()}",
                    gender=gender,
                    distance=int(distance),
                    stroke=stroke.strip(),
                    records=extract_records(section),
                    entries=parse_entries(section),
                    finals=[]  # Parse finals if needed
                )
                
                events.append(event)
                processed_events.add(event_key)
                
            except Exception as e:
                print(f"Error processing event in {pdf_path}: {e}")
                
    return events

def save_to_csv(events: List[SwimEvent], output_path: Path) -> None:
    """Save extracted events to CSV file."""
    rows = []
    
    for event in events:
        # Add entry for event even if no results
        base_row = {
            'pdf_path': event.pdf_path,
            'meet_name': event.meet_name,
            'meet_date': event.meet_date,
            'meet_location': event.meet_location,
            'event_number': event.event_number,
            'event_name': event.event_name,
            'gender': event.gender,
            'distance': event.distance,
            'stroke': event.stroke
        }
        
        # Add records if present
        for record_type, details in event.records.items():
            base_row[f'{record_type}_record'] = details.get('time')
            base_row[f'{record_type}_holder'] = details.get('holder')
        
        # Add entries
        if not event.entries:
            rows.append(base_row)
        else:
            for entry in event.entries:
                row = base_row.copy()
                row.update({
                    'place': entry.get('place'),
                    'athlete_name': entry.get('name'),
                    'age': entry.get('age'),
                    'team': entry.get('team'),
                    'time': entry.get('time')
                })
                rows.append(row)
    
    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

def main():
    """Process single PDF with complete event information."""
    base_dir = Path(__file__).parent.parent
    pdf_path = base_dir / "data" / "raw" / "Layout A" / "2025-pss-westmont-complete-meet-results.pdf"
    output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track exact event duplicates
    processed_events = set()
    all_events = []
    
    print(f"\nProcessing {pdf_path.name}...")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
            meet_info = extract_meet_info(text)
            
            # Split into event sections
            sections = re.split(r'(?:Event|EVENT)\s*(?::|#|\s)\s*\d+', text)
            
            for i, section in enumerate(sections[1:], 1):
                try:
                    # Print raw section for debugging
                    print(f"\n{'='*50}")
                    print(f"Processing section {i}:")
                    print(section[:200])  # Print first 200 chars
                    
                    # Parse event details
                    event_match = re.search(
                        r'(?:^|\n)\s*(Men|Women).*?(\d+)\s*(?:LC\s*)?(?:Meter|Meters)?\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)',
                        section,
                        re.IGNORECASE
                    )
                    
                    if not event_match:
                        print("No event match found, skipping section")
                        continue
                    
                    gender, distance, stroke = event_match.groups()
                    stroke = stroke.strip().upper()
                    
                    # Only skip if exact duplicate (including stroke)
                    event_key = (int(distance), stroke)
                    if event_key in processed_events:
                        print(f"Skipping duplicate: {distance} {stroke}")
                        continue
                    
                    # Extract records with debug output
                    records = {}
                    record_patterns = {
                        'world': r'(?:WORLD|World Record|WR)[:\s]+([0-9:\.]+)',
                        'american': r'(?:AMERICAN|American Record|AR)[:\s]+([0-9:\.]+)',
                        'us_open': r'(?:US OPEN|US Open Record|UO)[:\s]+([0-9:\.]+)'
                    }
                    
                    print("\nLooking for records:")
                    for record_type, pattern in record_patterns.items():
                        if match := re.search(pattern, section, re.IGNORECASE):
                            time = clean_time(match.group(1))
                            records[record_type] = time
                            print(f"Found {record_type} record: {time}")
                    
                    # Parse entries with improved pattern
                    entries = []
                    finals = []
                    entry_pattern = (
                        r'^\s*(\d+)\s+'  # Place
                        r'([A-Za-z\',\s-]+?)\s+'  # Name
                        r'(\d+)\s+'  # Age
                        r'([A-Z-]+)\s+'  # Team
                        r'([0-9:\.]+)'  # Entry time
                        r'(?:\s+(?:[0-9:\.]+\s+)*'  # Optional splits
                        r'([0-9:\.]+))?'  # Final time (optional)
                    )
                    
                    print("\nParsing entries:")
                    for line in section.split('\n'):
                        line = line.strip()
                        if line and line[0].isdigit():
                            if match := re.match(entry_pattern, line):
                                place, name, age, team, entry_time, final_time = match.groups()
                                print(f"Found entry: {place} {name.strip()} {entry_time}")
                                
                                result = {
                                    'place': int(place),
                                    'name': name.strip(),
                                    'age': int(age),
                                    'team': team,
                                    'entry_time': clean_time(entry_time),
                                    'final_time': clean_time(final_time) if final_time else None
                                }
                                
                                entries.append(result)
                                if final_time:
                                    finals.append(result)
                    
                    print(f"\nEvent summary:")
                    print(f"Event {i}: {gender} {distance} {stroke}")
                    print(f"Records found: {list(records.keys())}")
                    print(f"Entries: {len(entries)}, Finals: {len(finals)}")
                    
                    event = SwimEvent(
                        pdf_path=str(pdf_path),
                        meet_name=meet_info.get('name', ''),
                        meet_date=meet_info.get('date'),
                        meet_location=meet_info.get('location'),
                        event_number=i,
                        event_name=f"{distance} {stroke}",
                        gender=gender.upper(),
                        distance=int(distance),
                        stroke=stroke,
                        records=records,
                        entries=entries,
                        finals=finals
                    )
                    
                    all_events.append(event)
                    processed_events.add(event_key)
                    
                except Exception as e:
                    print(f"Error processing event: {str(e)}")
                    continue
            
        if all_events:
            output_path = output_dir / "westmont_2025_events.csv"
            rows = []
            
            for event in all_events:
                # If no entries, add event with just records
                if not event.entries:
                    row = {
                        'pdf_path': event.pdf_path,
                        'meet_name': event.meet_name,
                        'meet_date': event.meet_date,
                        'meet_location': event.meet_location,
                        'event_number': event.event_number,
                        'event_name': event.event_name,
                        'gender': event.gender,
                        'distance': event.distance,
                        'stroke': event.stroke,
                        'world_record': event.records.get('world'),
                        'american_record': event.records.get('american'),
                        'us_open_record': event.records.get('us_open')
                    }
                    rows.append(row)
                else:
                    # Add row for each entry
                    for entry in event.entries:
                        row = {
                            'pdf_path': event.pdf_path,
                            'meet_name': event.meet_name,
                            'meet_date': event.meet_date,
                            'meet_location': event.meet_location,
                            'event_number': event.event_number,
                            'event_name': event.event_name,
                            'gender': event.gender,
                            'distance': event.distance,
                            'stroke': event.stroke,
                            'world_record': event.records.get('world'),
                            'american_record': event.records.get('american'),
                            'us_open_record': event.records.get('us_open'),
                            'place': entry['place'],
                            'athlete_name': entry['name'],
                            'age': entry['age'],
                            'team': entry['team'],
                            'entry_time': entry['entry_time'],
                            'final_time': entry['final_time']
                        }
                        rows.append(row)
            
            pd.DataFrame(rows).to_csv(output_path, index=False)
            print(f"\nProcessed {len(all_events)} events")
            print(f"Results saved to {output_path}")
        else:
            print("\nNo events were processed successfully")
            
    except Exception as e:
        print(f"Failed to process PDF: {str(e)}")

if __name__ == "__main__":
    main()