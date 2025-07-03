import os
import logging
import time
import requests
from typing import Optional, Dict

import config

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt=config.DATE_FORMAT
    )

def read_csv(path):
    import pandas as pd
    return pd.read_csv(path)

def write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def read_parquet(path):
    import pandas as pd
    return pd.read_parquet(path)

def write_parquet(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def http_get_with_retries(
    url: str, 
    max_retries: int = 3, 
    headers: Optional[Dict[str, str]] = None
) -> requests.Response:
    default_headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    # Merge default headers with custom headers
    request_headers = default_headers.copy()
    if headers:
        request_headers.update(headers)
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=request_headers, timeout=10)
            response.raise_for_status()
            return response
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
            if attempt + 1 == max_retries:
                logging.error(f"All {max_retries} attempts failed for {url}")
                raise
            time.sleep(2 ** attempt)  # Exponential backoff