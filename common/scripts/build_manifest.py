import os
import csv
import soundfile as sf 
from pathlib import Path
import yaml
import re


# Columns in the final CSV
columns = [
    "filepath",
    "dataset",
    "speaker_id",
    "actor_gender",
    "native_label",
    "canonical_label",
    "native_intensity",
    "canonical_intensity",
    "duration_sec"
]

RAVDESS_DIR = Path("../../audio/datasets/ravdess")
CREMAD_DIR = Path("../../audio/datasets/crema_d/CREMA-D/AudioWAV")
IEMOCAP_DIR = Path("../../audio/datasets/iemocap/IEMOCAP_full_release/")
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
                native_intensity = _config[f"{dataset}_intensity_native"][parts[3]]
                canonical_intensity = _config[f"{dataset}_intensity_canonical"][parts[3]] 
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
                    "native_intensity": native_intensity,
                    "canonical_intensity" : canonical_intensity,
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
                native_intensity = _config[f"{dataset}_intensity_native"][parts[3]]
                canonical_intensity = _config[f"{dataset}_intensity_canonical"][parts[3]] 
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
                    "native_intensity": native_intensity,
                    "canonical_intensity" : canonical_intensity,
                    "duration_sec": duration_sec
                }
            )

    if skipped:
        print(f"[crema_d] skipped {len(skipped)} files:")
        for fp, err in skipped:
            print(f"  {fp} -> {err}")

    return rows


def parse_iemocap(root_dir, _config):
    rows = []
    skipped = []
    dataset = "iemocap"

    for i in range(1 , 6):

        eval_dir = root_dir / f"Session{i}" / "dialog" / "EmoEvaluation"  # one Session at a time, or loop Sessions outside

        for txt_file in eval_dir.glob("*.txt"):
            dialog_id = txt_file.stem  # e.g. "Ses02F_impro01"
            utt_labels = parse_emo_evaluation_file(txt_file)

            wav_dir = root_dir / f"Session{i}" / "sentences" / "wav" / dialog_id

            for utt_id, emo_code in utt_labels.items():
                wav_path = wav_dir / f"{utt_id}.wav"
                if not wav_path.exists():
                    skipped.append((str(wav_path), "wav not found"))
                    continue

                # derive speaker_id from session number + gender suffix on utt_id
                session_num = dialog_id[3:5]          # "02"
                speaker_gender = utt_id.split("_")[-1][0]  # "F" or "M" from e.g. "F000"
                speaker_id = f"Ses{session_num}_{speaker_gender}"

                try:
                    native_label = _config[f"{dataset}_native"][emo_code]
                    canonical_label = _config[dataset][emo_code]
                    duration_sec = get_duration(str(wav_path))
                except Exception as e:
                    skipped.append((str(wav_path), str(e)))
                    continue

                if canonical_label is None:
                    skipped.append((str(wav_path), f"dropped label: {emo_code}"))
                    continue
            
                rows.append({
                    "filepath": str(wav_path),
                    "dataset": dataset,
                    "speaker_id": speaker_id,
                    "actor_gender": "female" if speaker_gender == "F" else "male",
                    "native_label": native_label,
                    "canonical_label": canonical_label,
                    "native_intensity": None,     # IEMOCAP doesn't have RAVDESS/CREMA-D-style intensity tags
                    "canonical_intensity": None,
                    "duration_sec": duration_sec,
                })


    if skipped:
        print(f"[iemocap] skipped {len(skipped)} files")
        for fp, err in skipped[:10]:
            print(f"  {fp} -> {err}")

    return rows

def parse_emo_evaluation_file(txt_path):
    """
        Parse one EmoEvaluation .txt file, return dict: utterance_id -> emotion_code
    """
    utt_to_label = {}
    pattern = re.compile(r"^\[.*?\]\s+(\S+)\s+(\w+)\s+\[")
    with open(txt_path, "r") as f:
        for line in f:
            match = pattern.match(line)
            if match:
                utt_id, emo_code = match.groups()
                utt_to_label[utt_id] = emo_code
    return utt_to_label


def get_duration(filepath):
    info = sf.info(filepath)
    return info.frames / info.samplerate


def main(mapping_path, data1, data2, data3):
    with open(mapping_path, "r") as f:
        _config = yaml.safe_load(f)

    ravdess_rows = parse_ravdess(data1, _config)
    crema_d_rows = parse_cremad(data2, _config)
    iemocap_rows = parse_iemocap(data3, _config)

    manifest_path = Path(__file__).parent.parent / "manifests" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ravdess_rows)
        writer.writerows(crema_d_rows)
        writer.writerows(iemocap_rows)

    total = len(ravdess_rows) + len(crema_d_rows) + len(iemocap_rows)
    print(f"manifest written to {manifest_path}")
    print(f"total rows: {total} (ravdess={len(ravdess_rows)}, crema_d={len(crema_d_rows)}, iemocap={len(iemocap_rows)})")


if __name__ == "__main__":
    main(LABEL_MAPPING_DIR, RAVDESS_DIR, CREMAD_DIR, IEMOCAP_DIR)
