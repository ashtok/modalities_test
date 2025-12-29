from pathlib import Path

RAW_DIR = Path("data/raw")      # adjust relative path from src/
OUTPUT_DIR = Path("data/merged")  # folder for merged JSONL files
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File paths
train_files = {
    "train1_deu.jsonl": ["deu_Latn_sample_small.jsonl"],
    "train2_deu_eng.jsonl": ["deu_Latn_sample_small.jsonl", "eng_Latn_sample_small.jsonl"],
    "train3_all.jsonl": [
        "deu_Latn_sample_small.jsonl",
        "eng_Latn_sample_small.jsonl",
        "fra_Latn_sample_small.jsonl",
        "spa_Latn_sample_small.jsonl",
        "ita_Latn_sample_small.jsonl"
    ]
}

def merge_files(output_name, input_files):
    output_path = OUTPUT_DIR / output_name
    print(f"Creating {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as outfile:
        for fname in input_files:
            input_path = RAW_DIR / fname
            with open(input_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)
    print(f"Done: {output_path}")

def main():
    for out_name, files in train_files.items():
        merge_files(out_name, files)

if __name__ == "__main__":
    main()
