import subprocess
import sys
from pathlib import Path

def run_script(script_name):
    script_path = Path(__file__).parent / script_name
    print(f"Running {script_name}...")
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    """Main function to run the manual entry pipeline"""
    print("Manual Entry Pipeline")
    
    # Step 1: Convert CSV files to txt files
    print("\nStep 1: Converting CSV files to txt files for manual editing...")
    if not run_script("write_txts.py"):
        print("Failed to convert CSV files to txt files!")
        return
    
    # Step 2: Wait for user to edit txt files
    print("STEP 2: MANUAL EDITING")
    print("The txt files have been created in the 'txts_to_update' directory.")
    print("Please edit the txt files as needed.")
    print("\nWhen you're done editing, press Enter to continue...")
    input()
    
    # Step 3: Convert txt files back to CSV
    print("\nStep 3: Converting edited txt files back to CSV format...")
    if not run_script("read_txts.py"):
        print("Failed to convert txt files back to CSV!")
        return
    

    print("All files have been processed successfully.")

if __name__ == "__main__":
    main() 