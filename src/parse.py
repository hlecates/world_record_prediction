import pdfplumber
import re
import pandas as pd
from pathlib import Path
import os

def parse_meet_text(text: str) -> list:
    """Parse meet text into list of event dictionaries."""
    events_dict = {}  # Use dict to track events by number
    current_event_num = None
    current_records = []
    current_results = []

    # Regex patterns
    event_re = re.compile(r'^Event\s+(\d+)\s+(Women|Men)\s+(\d+)\s+LC\s+Meter\s+(.+)')
    # e.g. "World: 15:20.48 W 5/16/2018 Katie Ledecky USA"
    record_re = re.compile(
        r'^(World|American|U\.S\. Open):\s+'               # record type
        r'([\d:]+\.\d{2})\s+'                             # record time (e.g. 15:20.48)
        r'([A-Za-z])\s+'                                  # code (W, A, O, etc.)
        r'(\d{1,2}/\d{1,2}/\d{4})\s+'                    # date (MM/DD/YYYY)
        r'([A-Za-z\'-]+)'                                 # first name
        r'\s+'                                            # space between names
        r'([A-Za-z\'-]+)'                                 # last name
        r'\s+'                                            # space before team
        r'([A-Z][A-Za-z\s-]*?)$'                         # team/affiliation
    )
    # e.g. "3 Abby Dunford 19 Sarasota Sharks-FL 16:59.59 16:41.73"
    result_re = re.compile(
        r'^(\d+)\s+'                   # rank (one or more digits)
        r'(.+?)\s+'                    # name (non-greedy until next token)
        r'(\d{1,2})\s+'               # age (1–2 digits)
        r'(.+?)\s+'                   # team (non-greedy until time)
        r'((?:\d{1,2}:)?\d{2}\.\d{2})\s+'  # seed time (optional MM:)SS.xx
        r'((?:\d{1,2}:)?\d{2}\.\d{2})$'    # final time (optional MM:)SS.xx
    )

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Check for new event
        if event_match := event_re.match(line):
            event_num = event_match.group(1)
            
            # Save current records and results to previous event
            if current_event_num is not None:
                if current_event_num not in events_dict:
                    events_dict[current_event_num] = {
                        'event': current_event,
                        'records': current_records,
                        'results': current_results
                    }
                else:
                    events_dict[current_event_num]['results'].extend(current_results)
            
            # Start new event or continue existing one
            current_event_num = event_num
            current_event = line
            # Initialize empty records for new event, keep existing records for continuing event
            current_records = [] if event_num not in events_dict else events_dict[event_num]['records']
            current_results = []
            continue

        # Process records and results for current event
        if current_event_num:
            if m_rec := record_re.match(line):
                # Always add records (removed the if not current_records check)
                rec_type, rec_time, code, date, first_name, last_name, team = m_rec.groups()
                current_records.append({
                    'type': rec_type,
                    'time': rec_time,
                    'code': code,
                    'date': date,
                    'athlete': f"{first_name} {last_name}",
                    'team': team.strip()
                })
            
            elif m_res := result_re.match(line):
                rank, name, age, team, seed, final = m_res.groups()
                current_results.append({
                    'rank': int(rank),
                    'name': name.strip(),
                    'age': int(age),
                    'team': team.strip(),
                    'seed_time': seed,
                    'final_time': final
                })

    # Save final event
    if current_event_num is not None:
        if current_event_num not in events_dict:
            events_dict[current_event_num] = {
                'event': current_event,
                'records': current_records,
                'results': current_results
            }
        else:
            events_dict[current_event_num]['results'].extend(current_results)

    # Convert dict to list and return
    return list(events_dict.values())


def save_meet_data(meet_data: dict, output_path: Path, meet_name: str) -> None:
    """Save meet data in structured format with events, records, and entries."""
    
    # Extract event details
    event_match = re.search(r'Event\s+(\d+)\s+(Women|Men)\s+(\d+)\s+LC\s+Meter\s+(.+)', meet_data['event'])
    if not event_match:
        return
        
    event_num, gender, distance, stroke = event_match.groups()
    
    # Create structured event data
    event_data = {
        'event': event_num,
        'meet': meet_name,  # Use the passed meet name instead of hardcoded value
        'stroke': stroke.strip(),
        'gender': gender,
        'distance': int(distance),
        'records': [
            [record['type'], record['time'], record['date'], 
             record['athlete'], record['team']] for record in meet_data['records']
        ],
        'entries': [
            [result['rank'], result['name'], result['age'], 
             result['team'], result['seed_time'], result['final_time']] 
             for result in meet_data['results']
        ]
    }
    
    # Save as CSV with proper structure
    df = pd.DataFrame([event_data])
    if output_path.exists():
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_path, index=False)

def parse_pdf(pdf_path: str, output_path: Path) -> None:
    """Parse PDF and save structured event data."""
    # Extract meet name from PDF filename
    meet_name = Path(pdf_path).stem.replace('-complete-results', '').replace('-', ' ').title()
    
    # Extract text from PDF
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    
    # Parse all events
    events = parse_meet_text(all_text)
    
    # Clear existing file if it exists
    if output_path.exists():
        output_path.unlink()
    
    # Save each event with the extracted meet name
    for event in events:
        save_meet_data(event, output_path, meet_name)

def main():
    # Set up directory paths
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "data" / "raw" / "Layout A"
    output_dir = base_dir / "data" / "processed" / "parsed_meet_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all PDF files in Layout A directory
    for pdf_file in results_dir.glob("*.pdf"):
        # Generate output filename by replacing .pdf with .csv
        output_filename = pdf_file.stem + "_results.csv"
        output_path = output_dir / output_filename
        
        print(f"Processing: {pdf_file.name}")
        try:
            parse_pdf(str(pdf_file), output_path)
            print(f"Results saved to: {output_path}")
        except Exception as e:
            print(f"ERROR processing {pdf_file.name}: {str(e)}")

if __name__ == "__main__":
    main()