import os
import csv
import soundfile as sf 
from pathlib import Path
import yaml


# Columns in the final CSV
columns = [
    "filepath",
    "dataset",
    "speaker_id",
    "actor_gender",
    "native_label",
    "canonical_label",
    "intensity",
    "duration_sec"
]

RAVDESS_DIR = Path("../../audio/datasets/ravdess")
CREMAD_DIR = Path("../../audio/datasets/crema_d/CREMA-D/AudioWAV")
LABEL_MAPPING_DIR = Path("../configs/label_mapping.yaml")

def parse_ravdess(root_dir, _config):
    rows = []
    skipped = []
    dataset = "ravdess"

    for root, _, files in os.walk(root_dir):
        for filename in files:
            if not filename.lower().endswith(".wav"):   # guard against stray non-audio files
                continue

            parts = filename.replace(".wav", "").split("-")
            filepath = os.path.join(root, filename)

            try:
                native_label = _config[f"{dataset}_native"][parts[2]]
                canonical_label = _config[dataset][parts[2]]
                intensity = parts[3]
                speaker_id = int(parts[6])
                actor_gender = "female" if speaker_id % 2 == 0 else "male"
                duration_sec = get_duration(filepath=filepath)
            except Exception as e:
                skipped.append((filepath, str(e)))   # log and move on instead of crashing the whole run
                continue

            rows.append(
                {
                    "filepath": filepath,
                    "dataset": dataset,
                    "speaker_id": speaker_id,
                    "actor_gender": actor_gender,
                    "native_label": native_label,
                    "canonical_label": canonical_label,
                    "intensity": intensity,
                    "duration_sec": duration_sec
                }
            )

    if skipped:
        print(f"[ravdess] skipped {len(skipped)} files:")
        for fp, err in skipped:
            print(f"  {fp} -> {err}")

    return rows

def parse_cremad(root_dir, _config):
    rows = []
    skipped = []
    dataset = 'crema_d'

    for root, _, files in os.walk(root_dir):
        for filename in files:
            if not filename.lower().endswith(".wav"):
                continue

            parts = filename.replace(".wav", "").split("_")
            filepath = os.path.join(root, filename)

            try:
                speaker_id = int(parts[0])
                native_label = _config[f"{dataset}_native"][parts[2]]
                canonical_label = _config[dataset][parts[2]]
                intensity = parts[3]
                actor_gender = "NA"
                duration_sec = get_duration(filepath=filepath)
            except Exception as e:
                skipped.append((filepath, str(e)))
                continue

            rows.append(
                {
                    "filepath": filepath,
                    "dataset": dataset,
                    "speaker_id": speaker_id,
                    "actor_gender": actor_gender,
                    "native_label": native_label,
                    "canonical_label": canonical_label,
                    "intensity": intensity,
                    "duration_sec": duration_sec
                }
            )

    if skipped:
        print(f"[crema_d] skipped {len(skipped)} files:")
        for fp, err in skipped:
            print(f"  {fp} -> {err}")

    return rows

def get_duration(filepath):
    info = sf.info(filepath)
    return info.frames / info.samplerate


def main(mapping_path, data1, data2):
    with open(mapping_path, "r") as f:
        _config = yaml.safe_load(f)

    ravdess_rows = parse_ravdess(data1, _config)
    crema_d_rows = parse_cremad(data2, _config)

    manifest_path = Path(__file__).parent.parent / "manifests" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)   # create common/manifests/ if it doesn't exist yet

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ravdess_rows)
        writer.writerows(crema_d_rows)

    print(f"manifest written to {manifest_path}")
    print(f"total rows: {len(ravdess_rows) + len(crema_d_rows)} (ravdess={len(ravdess_rows)}, crema_d={len(crema_d_rows)})")


if __name__ == "__main__":
    main(LABEL_MAPPING_DIR, RAVDESS_DIR, CREMAD_DIR)
