import pandas as pd
import numpy as np
from pathlib import Path

MANIFEST_PATH = Path("../manifests/manifest.csv")
OUTPUT_DIR = Path("../manifests")
SEED = 42

# speaker counts per dataset (not the number of reacoding, rather number of actors/speakers)
SPLIT_RATIOS = {
    "ravdess":  {"train": 17, "val": 4, "test": 3},
    "crema_d":  {"train": 64, "val": 14, "test": 13},
    "iemocap":  {"train": 6, "val": 2, "test": 2},
}

def split_dataset(df, dataset_name, seed):
    # get unique speaker_ids, shuffle with seed
    # assign first N_train speakers -> train, next N_val -> val, rest -> test
    # return df with a new "split" column
    
    unique_speakers = df["speaker_id"].unique()
    unique_speakers = np.array(unique_speakers, dtype=object)  # force plain numpy array before shuffling, 
                                                               #Due to IEMOCAP's speaker id's being string
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_speakers)
    
    curr_split = SPLIT_RATIOS[dataset_name]
    speaker_to_split = {}
    offset = 0
    for split_name, count in curr_split.items():
        for speaker in unique_speakers[offset:offset + count]:
            speaker_to_split[speaker] = split_name
        offset += count

    df = df.copy()
    df["split"] = df["speaker_id"].map(speaker_to_split)
    assert df["split"].isna().sum() == 0, f"unmapped speakers in {dataset_name}"

    return df

def main():
    df = pd.read_csv(MANIFEST_PATH)

    splits = []
    for dataset in SPLIT_RATIOS:
        subset = df[df["dataset"] == dataset].copy()
        result = split_dataset(subset, dataset, SEED)
        splits.append(result)

    full = pd.concat(splits, ignore_index=True)
    
    for split_name in ["train", "val", "test"]:
        out = full[full["split"] == split_name]
        out.to_csv(OUTPUT_DIR / f"{split_name}.csv", index=False)
        print(f"{split_name}: {len(out)} clips")

    # also print per-dataset per-split breakdown
    print("\nbreakdown:")
    print(full.groupby(["dataset", "split"]).size().unstack(fill_value=0))
    print(full.groupby(["split", "canonical_label"]).size().unstack(fill_value=0))

if __name__ == "__main__":
    main()
