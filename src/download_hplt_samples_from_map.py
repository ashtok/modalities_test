import subprocess
from pathlib import Path
import requests

# Languages to process
LANGUAGES = [
    "deu_Latn",
    "eng_Latn",
    "fra_Latn",
    "spa_Latn",
    "ita_Latn",
]

BASE_URL = "https://data.hplt-project.org/three/sorted"
RAW_DIR = Path("data/raw")  # target folder for processed files
NUM_LINES = 500_000
TARGET_INDEX = 2  # pick the third file (0-based index)


def run(cmd):
    """Run a shell command and print it."""
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def fetch_map(lang):
    """Download the .map file and return a list of shard URLs."""
    map_url = f"{BASE_URL}/{lang}.map"
    print(f"Fetching map: {map_url}")
    response = requests.get(map_url)
    response.raise_for_status()

    # Each line in the .map file is already a complete URL
    urls = [line.strip() for line in response.text.splitlines() if line.strip()]
    return urls


def main():
    # Ensure target folder exists
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for lang in LANGUAGES:
        print(f"\nProcessing language: {lang}")
        if lang == "deu_Latn":
		    num_lines = 1_500_000
	    else:
		    num_lines = 1_500_000
        
	shard_urls = fetch_map(lang)

        if len(shard_urls) <= TARGET_INDEX:
            raise RuntimeError(f"Not enough shards for {lang}")

        shard_url = shard_urls[TARGET_INDEX]
        
        # Extract just the filename from the URL
        shard_filename = shard_url.split('/')[-1]
        
        # Include language in the temporary filename
        zst_path = RAW_DIR / f"{lang}_{shard_filename}"
        output_path = RAW_DIR / f"{lang}_sample_small.jsonl"

        # Download and process in one step (stream decompression, don't save full file)
        print(f"Downloading and extracting first {NUM_LINES} lines from {shard_url}")
        run([
            "bash",
            "-c",
            f"wget -qO- {shard_url} | zstd -dc | head -n {NUM_LINES} > {output_path}"
        ])

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
