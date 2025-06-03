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
class Event:
    id: str
    distance: int
    stroke: str
    records: List[Dict[str, str]]  # List of records for this event
    entries: List[Dict[str, str]]

all_events = {
    "50 Freestyle": False,
    "100 Freestyle": False,
    "200 Freestyle": False,
    "400 Freestyle": False,
    "800 Freestyle": False,
    "1500 Freestyle": False,    
    "100 Backstroke": False,
    "200 Backstroke": False,    
    "100 Breaststroke": False,
    "200 Breaststroke": False,
    "100 Butterfly": False,
    "200 Butterfly": False,
    "200 Individual Medley": False,
    "400 Individual Medley": False,  
}  


def main():
    

if __name__ == "__main__":
    main()