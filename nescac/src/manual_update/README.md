# Manual Entry Scripts

This directory contains scripts for manually editing swimming meet data by converting between CSV and text formats.

## Scripts Overview

### 1. `write_txts.py`
Converts all CSV files in the `individual` directory to text files for manual editing.

**What it does:**
- Reads all `*_parsed.csv` files from `nescac/data/processed/parsed/individual/`
- Converts each CSV to a human-readable text format
- Saves text files as `*_event_results.txt` in `txts_to_update/` directory

**Usage:**
```bash
python write_txts.py
```

### 2. `read_txts.py`
Converts edited text files back to CSV format and deletes the text files.

**What it does:**
- Reads all `*_event_results.txt` files from `txts_to_update/`
- Parses the text format back to CSV structure
- Saves as `*_parsed.csv` in the individual directory
- Deletes the text files after successful conversion

**Usage:**
```bash
python read_txts.py
```

### 3. `run_manual_entry.py`
Runs the complete pipeline with user interaction.

**What it does:**
1. Runs `write_txts.py` to create text files
2. Waits for user to edit the text files
3. Runs `read_txts.py` to convert back to CSV

**Usage:**
```bash
python run_manual_entry.py
```

## Workflow

### Step 1: Generate Text Files
```bash
python write_txts.py
```

This creates text files like:
- `2002-NESCAC-MSD-Results_parsed_event_results.txt`
- `2003-NESCAC-MSD-Results_parsed_event_results.txt`
- etc.

### Step 2: Edit Text Files
Manually edit the text files in the `txts_to_update/` directory. The format is:
```
Event 1:
  Meet: 2002_Nescac_Msd_Results
  Stroke: Freestyle Relay
  Distance: 200
  Event Type: relay
  Results:
    Swimmer 1:
      name: Smith, John
      yr: SR
      school: Williams
      exhibition: False
      seed_time: None
      prelim_time: 1:30.45
      finals_time: 1:29.87
      prelim_rank: 1
      final_rank: 1
```

### Step 3: Convert Back to CSV
```bash
python read_txts.py
```

This will:
- Convert all edited text files back to CSV format
- Save them to the individual directory (overwriting originals)
- Delete the text files

## File Structure

```
nescac/src/manual_entry/
├── write_txts.py          # CSV → Text conversion
├── read_txts.py           # Text → CSV conversion  
├── run_manual_entry.py    # Complete pipeline
├── README.md              # This file
└── txts_to_update/        # Directory for text files
    ├── 2002-NESCAC-MSD-Results_parsed_event_results.txt
    ├── 2003-NESCAC-MSD-Results_parsed_event_results.txt
    └── ...
```

## Notes

- The text format is designed to be human-readable and editable
- All data types (strings, numbers, booleans, None) are preserved
- The pipeline maintains the original CSV structure
- Text files are automatically cleaned up after conversion
- If any errors occur during conversion, the script will report them

## Error Handling

- If a text file can't be parsed, the script will report the error and continue with other files
- Original CSV files are only overwritten after successful conversion
- The script creates backup-like behavior by only deleting text files after successful CSV creation 