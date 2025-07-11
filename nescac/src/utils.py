import os
import logging
import time
import requests
from typing import Optional, Dict
from datetime import datetime

import config

def setup_logging(log_dir: str = "logs", log_filename: Optional[str] = None):
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp if not provided
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"pipeline_run_{timestamp}.log"
    
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt=config.DATE_FORMAT
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Set levels for noisy libraries
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    logging.getLogger("pdfplumber").setLevel(logging.WARNING)
    
    # Log the setup
    logging.info(f"Logging initialized. Log file: {log_filepath}")
    
    return log_filepath


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