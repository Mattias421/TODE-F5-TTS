import os
import sys

sys.path.append(os.getcwd())

import json
from importlib.resources import files
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
from datasets.arrow_writer import ArrowWriter
import argparse
import requests

TACO_TEST_URL = (
    "https://raw.githubusercontent.com/NVIDIA/tacotron2/refs/heads/master/filelists/ljs_audio_text_test_filelist.txt"
)
TACO_VAL_URL = (
    "https://raw.githubusercontent.com/NVIDIA/tacotron2/refs/heads/master/filelists/ljs_audio_text_val_filelist.txt"
)
TACO_TRAIN_URL = (
    "https://raw.githubusercontent.com/NVIDIA/tacotron2/refs/heads/master/filelists/ljs_audio_text_train_filelist.txt"
)


def download_and_extract_ids(url):
    """Downloads a filelist from a URL and extracts file IDs."""
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    file_ids = set()
    for line in response.text.splitlines():
        parts = line.split("|")
        file_id = parts[0].split("/")[-1].replace(".wav", "")  # Extract file ID
        file_ids.add(file_id)
    return file_ids


def main(test_mode):
    train_result = []
    test_result = []
    train_duration_list = []
    test_duration_list = []
    text_vocab_set = set()

    # Download filelist IDs for splitting
    test_ids = download_and_extract_ids(TACO_TEST_URL)
    # val_ids = download_and_extract_ids(TACO_VAL_URL) # TODO figure out what to do with val split
    train_ids = download_and_extract_ids(TACO_TRAIN_URL)

    with open(meta_info, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            uttr, text, norm_text = line.split("|")
            norm_text = norm_text.strip()
            wav_path = Path(dataset_dir) / "wavs" / f"{uttr}.wav"
            duration = sf.info(wav_path).duration

            if duration < 0.4 or duration > 30:
                continue

            data_entry = {"audio_path": str(wav_path), "text": norm_text, "duration": duration}
            text_vocab_set.update(list(norm_text))

            if uttr in test_ids:
                test_result.append(data_entry)
                test_duration_list.append(duration)
            elif uttr in train_ids:
                train_result.append(data_entry)
                train_duration_list.append(duration)

    save_dir_local = save_dir

    # save preprocessed dataset to disk
    if not os.path.exists(f"{save_dir_local}"):
        os.makedirs(f"{save_dir_local}")
    print(f"\nSaving to {save_dir_local} ...")

    result = test_result if test_mode else train_result
    duration_list = test_duration_list if test_mode else train_duration_list

    with ArrowWriter(path=f"{save_dir_local}/raw.arrow") as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)

    # dup a json separately saving duration in case for DynamicBatchSampler ease
    with open(f"{save_dir_local}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # vocab map, i.e. tokenizer
    # add alphabets and symbols (optional, if plan to ft on de/fr etc.)
    with open(f"{save_dir_local}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--test", action="store_true", help="Enable test mode")
    args = parser.parse_args()

    tokenizer = "char"  # "pinyin" | "char"

    dataset_dir = os.path.join(os.environ.get("DATA"), "LJSpeech-1.1")
    dataset_name = f"LJSpeech_test_{tokenizer}" if args.test else f"LJSpeech_train_{tokenizer}"
    meta_info = os.path.join(dataset_dir, "metadata.csv")
    save_dir = str(files("f5_tts").joinpath("../../data")) + f"/{dataset_name}"
    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")

    main(args.test)
