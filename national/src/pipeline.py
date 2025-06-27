import os
import time
import logging
import re
import pandas as pd
import pdfplumber
from pathlib import Path
from typing import List, Dict, Tuple, Set
from bs4 import BeautifulSoup
import requests
import ast


import config
import utils


class MeetDataPipeline:
    def __init__(self, output_base: Path):
        self.output_base = output_base
        self.raw_pdf_dir = output_base / "raw" / "meet_pdfs"
        self.processed_dir = output_base / "processed" / "parsed"
        self.clean_dir = output_base / "processed" / "clean"
        
        # Create directories
        self.raw_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        utils.setup_logging()
    

    def fetch_page(self, url: str, pause: float = 1.0) -> BeautifulSoup:
        resp = utils.http_get_with_retries(url, headers={"User-Agent": config.USER_AGENT})
        time.sleep(pause)
        return BeautifulSoup(resp.text, "html.parser")
    

    def fetch_all_meet_slugs(self) -> List[str]:
        index_url = f"{config.USA_SWIMMING_BASE}/times/data-hub/meet-results"
        logging.info(f"Fetching meet slugs from: {index_url}")
        
        soup = self.fetch_page(index_url)
        tabs = soup.select(".usas-content-leftrailnavigationoption2-tab a")
    
        slugs = []
        for a in tabs:
            href = a["data-usas-href"]
            slug = href.rstrip("/").split("/")[-1]
            slugs.append(slug)
        
        logging.info(f"Found {len(slugs)} meet categories")
        return slugs
    

    def download_meet_pdfs(self, slugs: List[str]) -> List[Path]:
        downloaded_pdfs = []
        
        for slug in slugs:
            meet_url = f"{config.USA_SWIMMING_BASE}/times/data-hub/meet-results/{slug}"
            logging.info(f"Processing meet category: {slug}")
            
            try:
                soup = self.fetch_page(meet_url)
                
                # Create subdirectory for this meet category
                meet_dir = self.raw_pdf_dir / slug
                meet_dir.mkdir(parents=True, exist_ok=True)
                
                # Find and download all PDF links
                pdf_links = soup.select("a[href$='.pdf']")
                logging.info(f"  Found {len(pdf_links)} PDFs")
                
                for link in pdf_links:
                    pdf_url = link["href"]
                    
                    # Make URL absolute
                    if pdf_url.startswith("//"):
                        pdf_url = "https:" + pdf_url
                    elif pdf_url.startswith("/"):
                        pdf_url = config.USA_SWIMMING_BASE + pdf_url
                    
                    # Get filename and output path
                    fname = pdf_url.split("/")[-1]
                    out_path = meet_dir / fname

                    if fname == "OG2020-_SWM_B99_SWM-------------------------------.pdf":
                        logging.info(f"Hard skipping file")
                        continue
                    
                    if out_path.exists():
                        logging.info(f"Already have {fname}")
                        downloaded_pdfs.append(out_path)
                        continue
                    
                    try:
                        logging.info(f"Downloading {fname}")
                        resp = requests.get(pdf_url, headers={"User-Agent": config.USER_AGENT})
                        resp.raise_for_status()
                        
                        out_path.write_bytes(resp.content)
                        downloaded_pdfs.append(out_path)
                        
                        # Respect rate limiting
                        time.sleep(config.REQUEST_PAUSE)
                        
                    except Exception as e:
                        logging.error(f"Failed to download {fname}: {e}")
                        
            except Exception as e:
                logging.error(f"Failed to process {slug}: {e}")
        
        logging.info(f"Downloaded {len(downloaded_pdfs)} PDFs total")
        return downloaded_pdfs
    

    def parse_meet_text(self, text: str) -> List[Dict]:
        events_dict = {}
        current_event_num = None
        current_records = []
        current_results = []

        event_re = re.compile(r'^Event\s+(\d+)\s+(Women|Men)\s+(\d+)\s+LC\s+Meter\s+(.+)')
        
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
                        unique_records = self.deduplicate_records(current_records)
                        events_dict[current_event_num] = {
                            'event': current_event,
                            'records': unique_records,
                            'results': current_results
                        }
                    else:
                        events_dict[current_event_num]['results'].extend(current_results)
                
                # Start new event or continue existing one
                current_event_num = event_num
                current_event = line

                # Only reuse existing records when extending results, not when reprocessing
                if event_num not in events_dict:
                    current_records = []
                    current_results = []
                else:
                    # If event exists, don't reprocess records, just collect new results
                    current_records = []  # Don't reprocess records
                    current_results = []
                continue

            # Process records and results for current event
            if current_event_num:
                if m_rec := record_re.match(line):
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
            
            # Parse events using exact same logic
            events = self.parse_meet_text(all_text)
            
            # Extract meet name from filename
            meet_name = pdf_path.stem.replace('-complete-results', '').replace('-', ' ').title()
            
            # Add meet context to each event
            processed_events = []
            for event in events:
                # Extract event details
                event_match = re.search(r'Event\s+(\d+)\s+(Women|Men)\s+(\d+)\s+LC\s+Meter\s+(.+)', event['event'])
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
                        [record['type'], record['time'], record['date'], 
                         record['athlete'], record['team']] for record in event['records']
                    ],
                    'entries': [
                        [result['rank'], result['name'], result['age'], 
                         result['team'], result['seed_time'], result['final_time']] 
                         for result in event['results']
                    ]
                }
                processed_events.append(processed_event)
            
            logging.info(f"  Extracted {len(processed_events)} events, {sum(len(e['entries']) for e in processed_events)} total results")
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

        # Hardcoded formatting of strokes 
        cleaned_df['stroke'] = cleaned_df['stroke'].astype(str).str.replace(' Knockout', '', regex=False)
        cleaned_df['stroke'] = cleaned_df['stroke'].astype(str).str.replace('(cid:976)', 'f', regex=False)

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
        
        logging.info("Downloading meet PDFs")
        slugs = self.fetch_all_meet_slugs()
        pdf_paths = self.download_meet_pdfs(slugs)
        
        if not pdf_paths:
            logging.error("No PDFs downloaded")
            return None
        
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