import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transcribe import transcribe

PROJECT_ROOT  = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = PROJECT_ROOT / "common" / "manifests"

def main():
    test_df = pd.read_csv(MANIFESTS_DIR / "test.csv")

    transcripts = []
    skipped = 0

    for filepath in tqdm(test_df["filepath"], desc="transcribing"):
        try:
            text = transcribe(filepath)
            transcripts.append(text)
            if text is None:
                skipped += 1
        except Exception as e:
            print(f"\nfailed: {filepath} -> {e}")
            transcripts.append(None)
            skipped += 1

    test_df["transcript"] = transcripts

    out_path = MANIFESTS_DIR / "test_transcripts.csv"
    test_df.to_csv(out_path, index=False)

    print(f"\nsaved to {out_path}")
    print(f"total: {len(test_df)} | transcribed: {len(test_df) - skipped} | null/failed: {skipped}")

if __name__ == "__main__":
    main()
    