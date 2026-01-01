import subprocess
from pathlib import Path
import requests

# =====================
# CONFIG
# =====================

LANGUAGES = [
    "deu_Latn",
    "eng_Latn",
    "fra_Latn",
    "spa_Latn",
    "ita_Latn",
]

BASE_URL = "https://data.hplt-project.org/three/sorted"
TARGET_INDEX = 2  # 0-based index of shard

RAW_DIR = Path("data/raw")
MERGED_DIR = Path("data/merged")

RAW_DIR.mkdir(parents=True, exist_ok=True)
MERGED_DIR.mkdir(parents=True, exist_ok=True)

LINES_PER_LANG = {
    "deu_Latn": 1_200_000,
    "default": 600_000,
}

TRAIN_FILES = {
    "train1_deu.jsonl": [
        "deu_Latn_sample_small.jsonl",
    ],
    "train2_deu_eng.jsonl": [
        "deu_Latn_sample_small.jsonl",
        "eng_Latn_sample_small.jsonl",
    ],
    "train3_all.jsonl": [
        "deu_Latn_sample_small.jsonl",
        "eng_Latn_sample_small.jsonl",
        "fra_Latn_sample_small.jsonl",
        "spa_Latn_sample_small.jsonl",
        "ita_Latn_sample_small.jsonl",
    ],
}

# =====================
# HELPERS
# =====================

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def fetch_map(lang):
    map_url = f"{BASE_URL}/{lang}.map"
    print(f"Fetching map: {map_url}")
    r = requests.get(map_url)
    r.raise_for_status()
    return [line.strip() for line in r.text.splitlines() if line.strip()]


# =====================
# STEP 1: DOWNLOAD + SAMPLE
# =====================

def download_samples():
    for lang in LANGUAGES:
        print(f"\nProcessing language: {lang}")

        num_lines = LINES_PER_LANG.get(lang, LINES_PER_LANG["default"])
        shard_urls = fetch_map(lang)

        if len(shard_urls) <= TARGET_INDEX:
            raise RuntimeError(f"Not enough shards for {lang}")

        shard_url = shard_urls[TARGET_INDEX]
        output_path = RAW_DIR / f"{lang}_sample_small.jsonl"

        print(f"Downloading first {num_lines} lines from {shard_url}")
        run([
            "bash",
            "-c",
            f"wget -qO- {shard_url} | zstd -dc | head -n {num_lines} > {output_path}"
        ])

        print(f"Saved: {output_path}")


# =====================
# STEP 2: MERGE FILES
# =====================

def merge_files(output_name, input_files):
    output_path = MERGED_DIR / output_name
    print(f"Creating {output_path}")

    with open(output_path, "w", encoding="utf-8") as outfile:
        for fname in input_files:
            input_path = RAW_DIR / fname
            if not input_path.exists():
                raise FileNotFoundError(f"Missing input file: {input_path}")

            with open(input_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)

    print(f"Done: {output_path}")


def merge_all():
    for out_name, files in TRAIN_FILES.items():
        merge_files(out_name, files)


# =====================
# MAIN
# =====================

def main():
    download_samples()
    merge_all()


if __name__ == "__main__":
    main()
