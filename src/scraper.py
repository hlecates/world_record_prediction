import os
import time
import logging
import json
from typing import List, Dict
from bs4 import BeautifulSoup
from typing import Optional
from pathlib import Path

import config
import utils

def fetch_swimming_records() -> List[Dict]:
    """
    Fetch swimming world records using the World Aquatics API endpoint.
    """
    url = "https://www.worldaquatics.com/swimming/records"
    
    headers = {
        "User-Agent": config.USER_AGENT,
        "Accept": "application/json"
    }
    
    params = {
        "gender": "ALL",
        "distance": "50,100,200,400,800,1500",
        "stroke": "FREESTYLE,BACKSTROKE,BREASTSTROKE,BUTTERFLY,MEDLEY",
        "poolConfiguration": "LCM"
    }
    
    try:
        response = utils.http_get_with_retries(url, headers=headers)
        records = response.json()
        
        # Save raw response
        out_file = os.path.join(config.RECORDS_RAW_SUBDIR, "world_records.json")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, 'w') as f:
            json.dump(records, f, indent=2)
            
        return records
        
    except Exception as e:
        logging.error(f"Failed to fetch records: {str(e)}")
        return []

def save_raw_html(html: str, filepath: str) -> bool:
    """
    Save raw HTML content to file.
    
    Args:
        html: HTML content to save
        filepath: Path to save the file
    Returns:
        bool: True if save was successful
    """
    try:
        # Ensure directory exists
        Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        logging.info(f"Saved raw HTML to {filepath}")
        return True
    except Exception as e:
        logging.error(f"Failed to save HTML to {filepath}: {str(e)}")
        return False

def fetch_page(url: str, source: str = "default", pause: Optional[float] = None) -> BeautifulSoup:
    """
    Fetch a URL and return BeautifulSoup object with rate limiting.
    Args:
        url: URL to fetch
        source: Source site name for rate limiting
        pause: Optional override for pause duration
    """
    headers = {"User-Agent": config.USER_AGENT}
    resp = utils.http_get_with_retries(url, headers=headers)
    
    # Use source-specific pause if available
    pause_time = pause or config.REQUEST_PAUSE
    time.sleep(pause_time)
    return BeautifulSoup(resp.text, 'html.parser')

def download_world_records():
    """Download and save world records from World Aquatics."""
    logging.info(f"Downloading world records from {config.WORLD_RECORDS_URL}")
    try:
        soup = fetch_page(config.WORLD_RECORDS_URL, source="worldaquatics")
        out_file = os.path.join(config.RECORDS_RAW_SUBDIR, "world_records.html")
        save_raw_html(soup.prettify(), out_file)
        return True
    except Exception as e:
        logging.error(f"Failed to download world records: {str(e)}")
        return False

def download_swimcloud_meet(meet_id: str):
    """Download meet results from SwimCloud."""
    logging.info(f"Downloading SwimCloud meet {meet_id}")
    try:
        url = config.MEET_URL_TEMPLATE.format(meet_id=meet_id)
        soup = fetch_page(url, source="swimcloud")
        out_file = os.path.join(config.MEETS_RAW_SUBDIR, f"swimcloud_{meet_id}.html")
        save_raw_html(soup.prettify(), out_file)
        return True
    except Exception as e:
        logging.error(f"Failed to download SwimCloud meet {meet_id}: {str(e)}")
        return False

def download_usa_rankings(event: str):
    """Download USA Swimming rankings for an event."""
    logging.info(f"Downloading USA Swimming rankings for {event}")
    try:
        # Construct search parameters for USA Swimming database
        params = {
            "event": event,
            "gender": "ALL",
            "course": "LCM",
            "season": "2024"
        }
        url = f"{config.TIMES_DATABASE_URL}?{'&'.join(f'{k}={v}' for k,v in params.items())}"
        soup = fetch_page(url, source="usaswimming")
        out_file = os.path.join(config.RECORDS_RAW_SUBDIR, f"usa_rankings_{event}.html")
        save_raw_html(soup.prettify(), out_file)
        return True
    except Exception as e:
        logging.error(f"Failed to download USA rankings for {event}: {str(e)}")
        return False

def download_event_results(meet_id: str, event_id: str):
    """Download specific event results from SwimCloud."""
    logging.info(f"Downloading results for event {event_id} from meet {meet_id}")
    try:
        url = config.EVENT_URL_TEMPLATE.format(meet_id=meet_id, event_id=event_id)
        soup = fetch_page(url, source="swimcloud")
        out_file = os.path.join(
            config.MEETS_RAW_SUBDIR, 
            f"swimcloud_{meet_id}_event_{event_id}.html"
        )
        save_raw_html(soup.prettify(), out_file)
        return True
    except Exception as e:
        logging.error(f"Failed to download event results: {str(e)}")
        return False

def fetch_usa_meet_results_page() -> Optional[str]:
    """
    Fetch the raw HTML for the USA Swimming meet-results hub page.
    Returns the HTML string on success, or None on failure.
    """
    url = f"{config.USA_SWIMMING_BASE}/times/data-hub/meet-results"
    logging.info(f"Fetching USA Swimming meet-results page: {url}")
    try:
        # Use your utils retry wrapper
        resp = utils.http_get_with_retries(url, headers={"User-Agent": config.USER_AGENT})
        html = resp.text
        
        # Optionally save it to disk, too:
        out_file = os.path.join(config.RAW_DATA_DIR, "usaswimming_meet_results.html")
        save_raw_html(html, out_file)
        
        return html
    except Exception as e:
        logging.error(f"Failed to fetch USA meet-results HTML: {e}")
        return None
    
def fetch_all_usa_meets_index() -> List[str]:
    """
    Returns a list of slugs (e.g. 'olympic-trials-results', 'tyr-pro-swim-series-results', …)
    from the USA Swimming meet-results hub page.
    """
    url = f"{config.USA_SWIMMING_BASE}/times/data-hub/meet-results"
    soup = fetch_page(url, source="usaswimming")
    tabs = soup.select(".usas-content-leftrailnavigationoption2-tab a")
    slugs = []
    for a in tabs:
        href = a["data-usas-href"]  # e.g. "//www.usaswimming.org/times/data-hub/meet-results/tyr-pro-swim-series-results"
        # strip off the leading // and split
        clean = href.split("/")[-1]
        slugs.append(clean)
    return slugs


def download_usa_meet_pdfs():
    slugs = fetch_all_usa_meets_index()
    for slug in slugs:
        meet_url = f"{config.USA_SWIMMING_BASE}/times/data-hub/meet-results/{slug}"
        logging.info(f"→ fetching meet page: {meet_url}")
        soup = fetch_page(meet_url, source="usaswimming")
        # find all the PDF links on that page:
        for pdf in soup.select("a[href$='.pdf']"):
            pdf_url = pdf["href"]
            # make it absolute if necessary
            if pdf_url.startswith("//"):
                pdf_url = "https:" + pdf_url
            elif pdf_url.startswith("/"):
                pdf_url = config.USA_SWIMMING_BASE + pdf_url

            fname = pdf_url.split("/")[-1]
            out_path = os.path.join(config.MEETS_RAW_SUBDIR, slug, fname)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            logging.info(f"   downloading pdf {fname}")
            resp = utils.http_get_with_retries(pdf_url, headers={"User-Agent": config.USER_AGENT})
            with open(out_path, "wb") as f:
                f.write(resp.content)
            time.sleep(config.REQUEST_PAUSE)

if __name__ == "__main__":
    utils.setup_logging()
    html = fetch_usa_meet_results_page()
    if html:
        print("Fetched USA meet-results HTML (length:", len(html), "chars)")
    else:
        print("Could not fetch USA meet-results page.")