import subprocess
from pathlib import Path

# Paths
MERGED_DIR = Path("data/merged")
INDEX_DIR = Path("data/preprocessed")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    # Loop through all JSONL files in the raw folder
    for jsonl_file in MERGED_DIR.glob("*.jsonl"):
        # Build index filename
        index_file = INDEX_DIR / f"{jsonl_file.stem}.idx"

        # Run the modalities command
        run([
            "modalities",
            "data",
            "create_raw_index",
            "--index_path",
            str(index_file),
            str(jsonl_file)
        ])

        print(f"Created index: {index_file}")

if __name__ == "__main__":
    main()
