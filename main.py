import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import torch
import tiktoken
from pathlib import Path

from src.config import load_config
from src.train import run_training
from src.predict import run_inference
from src.evaluate import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="GPT-2 Language Classifier")
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to the configuration file"
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = tiktoken.get_encoding("gpt2")

    if cfg.meta.mode == "train":
        run_training(cfg, device, tokenizer)

    if cfg.meta.mode in ["train", "evaluate"]:
        run_evaluation(cfg, device, tokenizer)

    elif cfg.meta.mode == "predict":
        run_inference(cfg, device, tokenizer)

    else:
        print(f"Unknown mode. Use 'train' or 'predict'.")


if __name__ == "__main__":
    main()
