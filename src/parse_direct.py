from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import pdfplumber

@dataclass
class SwimEvent:
    """Structure for swim event data"""
    meet_name: str
    event_number: int 
    event_name: str
    gender: str
    distance: int
    stroke: str
    records: Dict[str, Dict[str, str]]  # {"world": {"time": "...", "holder": "...", "date": "..."}}
    entries: List[Dict]  # Trials/prelims results
    finals: List[Dict]   # Finals results if available

def extract_records(lines: List[str]) -> Dict[str, Dict[str, str]]:
    """Extract world, american, and other records."""
    records = {}
    record_types = ["WORLD RECORD:", "AMERICAN RECORD:", "US OPEN RECORD:", "POOL RECORD:"]
    
    for line in lines:
        for record_type in record_types:
            if record_type in line:
                parts = line.split(record_type)[1].strip().split()
                time = parts[0].strip('W').strip('A').strip('O')
                # Handle cases where holder name contains spaces
                holder = " ".join(parts[1:-1])
                date = parts[-1]
                
                key = record_type.replace(" RECORD:", "").strip().lower()
                records[key] = {
                    "time": time,
                    "holder": holder,
                    "date": date
                }
    return records

def parse_result_line(line: str, round_type: str) -> Optional[Dict]:
    """Parse a single result line."""
    try:
        parts = line.strip().split()
        if not parts[0][0].isdigit():
            return None
            
        result = {
            "place": parts[0].rstrip(')'),
            "name": " ".join(parts[1:-3]),
            "age": parts[-3],
            "team": parts[-2],
            "time": parts[-1],
            "round": round_type
        }
        return result
    except:
        return None

def parse_meet_file(filepath: Path) -> List[SwimEvent]:
    """Parse a meet results file into structured data."""
    events = []
    current_event = None
    lines = []
    
    # Read file content
    with open(filepath) as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for event start
        if "EVENT" in line or "Event" in line:
            # Save previous event if exists
            if current_event:
                events.append(current_event)
            
            # Parse event header
            event_num = int(line.split(':')[1].split()[0])
            event_name = " ".join(line.split(':')[1].split()[1:])
            gender = "WOMEN" if "WOMEN" in event_name else "MEN"
            
            # Find distance and stroke
            parts = event_name.split()
            distance = int(parts[2])
            stroke = " ".join(parts[3:]).replace("Meters", "").strip()
            
            # Get records section
            record_lines = []
            j = i + 1
            while j < len(lines) and not "TRIALS" in lines[j]:
                record_lines.append(lines[j])
                j += 1
            records = extract_records(record_lines)
            
            # Initialize new event
            current_event = SwimEvent(
                meet_name=filepath.parent.name,
                event_number=event_num,
                event_name=event_name,
                gender=gender,
                distance=distance,
                stroke=stroke,
                records=records,
                entries=[],
                finals=[]
            )
            
            # Skip to results
            i = j + 2
            continue
            
        # Parse results
        if current_event:
            result = parse_result_line(line, "TRIALS" if "TRIALS" in line else "FINALS")
            if result:
                if "FINAL" in line:
                    current_event.finals.append(result)
                else:
                    current_event.entries.append(result)
        
        i += 1
    
    # Add last event
    if current_event:
        events.append(current_event)
        
    return events

def save_to_structured_data(events: List[SwimEvent], output_dir: Path):
    """Save parsed events to CSV files for ML."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare DataFrames
    records_rows = []
    results_rows = []
    
    for event in events:
        # Add records
        for record_type, details in event.records.items():
            records_rows.append({
                "meet": event.meet_name,
                "event_number": event.event_number,
                "event_name": event.event_name,
                "gender": event.gender,
                "distance": event.distance,
                "stroke": event.stroke,
                "record_type": record_type,
                "time": details["time"],
                "holder": details["holder"],
                "date": details["date"]
            })
        
        # Add results
        for result in event.entries + event.finals:
            results_rows.append({
                "meet": event.meet_name,
                "event_number": event.event_number,
                "event_name": event.event_name,
                "gender": event.gender,
                "distance": event.distance,
                "stroke": event.stroke,
                **result
            })
    
    # Save to CSV
    pd.DataFrame(records_rows).to_csv(output_dir / "records.csv", index=False)
    pd.DataFrame(results_rows).to_csv(output_dir / "results.csv", index=False)

def main():
    # Configure paths
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "data" / "processed" / "meet_results"
    output_dir = base_dir / "data" / "ml_features"
    
    # Process each results file
    for meet_dir in results_dir.iterdir():
        if meet_dir.is_dir():
            for results_file in meet_dir.glob("*.txt"):
                print(f"Processing {results_file}")
                events = parse_meet_file(results_file)
                print(f"Found {len(events)} events")
                save_to_structured_data(events, output_dir)

if __name__ == "__main__":
    main()