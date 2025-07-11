import logging
import re
import pandas as pd
import pdfplumber
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from pdf_parser import PDFParser
from txt_parser import TXTParser
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

        # Setup logging and initialize parsers
        utils.setup_logging()
        self.pdf_parser = PDFParser()
        self.txt_parser = TXTParser()
        self.data_formatter = DataFormatter()

        # Store individual meet DataFrames in memory
        self.individual_meet_dfs = {}

    def parse_single_pdf(self, pdf_path: Path) -> List[Dict]:
        logging.info(f"Parsing PDF: {pdf_path.name}")
        try:
            all_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"

            events = self.pdf_parser.parse_meet_text(all_text)
            return self._process_events(events, pdf_path)

        except Exception as e:
            logging.error(f"Failed to parse {pdf_path.name}: {e}")
            return []

    def parse_single_txt(self, txt_path: Path) -> List[Dict]:
        logging.debug(f"Parsing TXT: {txt_path.name}")
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as file:
                all_text = file.read()

            events = self.txt_parser.parse_meet_text(all_text)
            return self._process_events(events, txt_path)

        except Exception as e:
            logging.error(f"Failed to parse {txt_path.name}: {e}")
            return []

    def _process_events(self, events: List[Dict], source_path: Path) -> List[Dict]:
        """Normalize event dicts into flat records with unified results."""
        meet_name = source_path.stem.replace('-complete-results', '').replace('-', ' ').title()
        source_file = source_path.name
        meet_category = source_path.parent.name

        processed_events = []
        for event in events:
            # Extract header info
            match = re.match(
                r'Event\s+(\d+)\s+(Women|Men)\s+(\d+)\s+Yard\s+(.+)',
                event.get('event', '')
            )
            if not match:
                continue
            event_num, gender, distance, stroke = match.groups()

            results = event.get('results', [])
            total_results = len(results)
            finals_results = sum(1 for r in results if r.get('finals_time'))
            
            # Debug information
            logging.debug(f"Event {event_num} {gender} {distance}Y {stroke.strip()}: "
                        f"{finals_results}/{total_results} results have finals times")

            base = {
                'event': event_num,
                'meet': meet_name,
                'stroke': stroke.strip(),
                'gender': gender,
                'distance': int(distance),
                'source_file': source_file,
                'meet_category': meet_category,
                'event_type': event.get('event_type', 'individual'),
                'results': results
            }

            processed_events.append(base)

        return processed_events

    def parse_all_files(
        self,
        pdf_paths: Optional[List[Path]] = None,
        txt_paths: Optional[List[Path]] = None,
        save_individual: bool = True
    ) -> pd.DataFrame:
        if pdf_paths is None:
            pdf_paths = list(self.raw_pdf_dir.rglob("*.pdf"))
        if txt_paths is None:
            txt_paths = list(self.raw_txt_dir.rglob("*.txt"))

        total = len(pdf_paths) + len(txt_paths)
        logging.debug(f"Parsing {len(pdf_paths)} PDF and {len(txt_paths)} TXT files ({total} total)")

        all_events = []
        success = 0
        individual_files = []

        for p in pdf_paths:
            ev = self.parse_single_pdf(p)
            if ev:
                all_events.extend(ev)
                if save_individual:
                    individual_files.append((p.stem, ev))
                success += 1
        for t in txt_paths:
            ev = self.parse_single_txt(t)
            if ev:
                all_events.extend(ev)
                if save_individual:
                    individual_files.append((t.stem, ev))
                success += 1

        logging.debug(f"Successfully parsed {success}/{total} files, extracted {len(all_events)} events")

        if save_individual and individual_files:
            self._save_individual_files(individual_files)

        if not all_events:
            logging.warning("No events were parsed!")
            return pd.DataFrame()
        return pd.DataFrame(all_events)

    def _save_individual_files(self, individual_files: List[Tuple[str, List[Dict]]]):
        """Save each meet's events to its own CSV file."""
        for meet_name, events in individual_files:
            if not events:
                continue
                
            df = pd.DataFrame(events)
            if df.empty:
                continue
                
            # Create a clean filename
            clean_name = meet_name.replace('_', '-').replace(' ', '-')
            output_path = self.processed_dir / "individual" / f"{clean_name}_parsed.csv"
            
            self.save_data(df, output_path)
            logging.info(f"Saved individual meet file: {output_path.name}")

    def save_data(self, df: pd.DataFrame, output_path: Path) -> Optional[Path]:
        if df.empty:
            logging.warning(f"No data to save for {output_path}")
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.debug(f"Saved data to {output_path}")
        return output_path

    def clean_existing_data(self) -> Tuple[Optional[Path], Optional[Path]]:
        logging.info("Cleaning existing parsed data")
        parsed = self.processed_dir / "parsed_events.csv"
        if not parsed.exists():
            logging.error(f"Parsed data not found at {parsed}")
            return None, None
        df = pd.read_csv(parsed)
        clean_df = self.data_formatter.clean_dataframe(df)
        clean_path = self.save_data(clean_df, self.clean_dir / "clean_events.csv")
        return parsed, clean_path

    def run_pipeline(self) -> Tuple[Optional[Path], Optional[Path]]:
        logging.info("Starting meet data pipeline")
        df = self.parse_all_files()
        parsed_path = self.save_data(df, self.processed_dir / "parsed_events.csv")
        clean_df = self.data_formatter.clean_dataframe(df)
        clean_path = self.save_data(clean_df, self.clean_dir / "clean_events.csv")
        return parsed_path, clean_path


def main():
    base = Path(__file__).parent.parent
    pipeline = MeetDataPipeline(base / "data")

    import sys
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else None
    if cmd == "--parse":
        df = pipeline.parse_all_files(save_individual=True)
        path = pipeline.save_data(df, pipeline.processed_dir / "parsed_events.csv")
        print(f"Success: {path}" if path else "Failed to save parsed data.")
        print("Individual meet files have been saved to the processed/parsed directory.")
    elif cmd == "--clean":
        parsed, clean = pipeline.clean_existing_data()
        print(f"Success: {clean}" if clean else "Failed to clean data.")
    else:
        parsed, clean = pipeline.run_pipeline()
        if parsed and clean:
            print(f"Success: Generated {parsed.name}, {clean.name}")
        elif parsed:
            print(f"Partial success: Generated {parsed.name}")
        else:
            print("Pipeline failed.")

if __name__ == "__main__":
    main()
