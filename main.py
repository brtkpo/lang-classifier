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
    """
    Main entry point for the GPT-2 Language Classifier workflow.

    This script orchestrates the full pipeline: training, standalone evaluation, 
    and interactive prediction, depending on the provided JSON configuration.

    Parameters
    ----------
    None

    Behavior
    --------
    - If `mode` in configuration is "train", trains the GPT-2 classifier model.
    - If `mode` is "evaluate" or after training, evaluates the model on the test dataset.
    - If `mode` is "predict", launches an interactive CLI to classify custom text.

    Notes
    -----
    Device selection (CPU or CUDA) is handled automatically.
    TensorFlow warnings are suppressed by default via environment variables.
    The script uses the 'tiktoken' library for accurate GPT-2 BPE tokenization.

    Example
    -------
    python main.py --config config.json
    """
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
