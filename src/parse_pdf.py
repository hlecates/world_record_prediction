import sys
from pathlib import Path
from typing import Dict, List, Optional
import pdfplumber
import pandas as pd

# Configure paths
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "meet_results"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "meet_results"

def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def find_pdfs(base_dir: Path) -> Dict[str, List[Path]]:
    """
    Find PDFs organized by meet subdirectory.
    Returns: Dict[meet_name, List[pdf_paths]]
    """
    pdfs_by_meet = {}
    for meet_dir in base_dir.iterdir():
        if meet_dir.is_dir():
            pdfs = list(meet_dir.glob("*.pdf"))
            if pdfs:
                pdfs_by_meet[meet_dir.name] = pdfs
    return pdfs_by_meet

def extract_text(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                text.append(f"--- PAGE {i} ---\n{page_text}\n")
        return "\n".join(text)
    except Exception as e:
        print(f"Failed to extract text from {pdf_path}: {e}")
        return ""

def save_text(text: str, output_path: Path) -> None:
    """Save extracted text to a file."""
    try:
        output_path.write_text(text, encoding='utf-8')
        print(f"Saved text to {output_path}")
    except Exception as e:
        print(f"Failed to save text to {output_path}: {e}")

def process_meet_pdfs(meet_name: str, pdf_paths: List[Path]) -> List[Dict]:
    """Process all PDFs for a specific meet."""
    results = []
    meet_output_dir = OUTPUT_DIR / meet_name
    meet_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(pdf_paths)} PDFs for meet: {meet_name}")
    
    for pdf_path in pdf_paths:
        try:
            # Create output path preserving filename
            output_path = meet_output_dir / f"{pdf_path.stem}.txt"
            
            # Extract and save text
            text = extract_text(pdf_path)
            if text:
                save_text(text, output_path)
                results.append({
                    'meet': meet_name,
                    'pdf_name': pdf_path.name,
                    'pdf_path': str(pdf_path),
                    'txt_path': str(output_path),
                    'size': len(text),
                    'status': 'success'
                })
        except Exception as e:
            print(f"Failed to process {pdf_path}: {e}")
            results.append({
                'meet': meet_name,
                'pdf_name': pdf_path.name,
                'pdf_path': str(pdf_path),
                'status': 'failed',
                'error': str(e)
            })
    
    return results


def main():
    """Main function to process all PDFs in the raw directory."""
    setup_directories()
    
    # Find all PDF files
    pdfs_by_meet = find_pdfs(RAW_DIR)
    if not pdfs_by_meet:
        print(f"No PDFs found in {RAW_DIR}")
        sys.exit(1)
    print(f"Found {len(pdfs_by_meet)} PDF files to process.")
    
    all_results = []
    for meet_name, pdf_paths in pdfs_by_meet.items():
        meet_results = process_meet_pdfs(meet_name, pdf_paths)
        all_results.extend(meet_results)
    
    # Save results to a CSV file
    results_df = pd.DataFrame(all_results)
    results_csv_path = OUTPUT_DIR / "pdf_processing_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved processing results to {results_csv_path}")
    print("PDF processing complete.")

if __name__ == "__main__":
    main()