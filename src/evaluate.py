import torch
import tiktoken
from torch.utils.data import DataLoader

from .config import Config
from .dataset import LanguageDataset
from .model import setup_model
from .train import calc_accuracy_loader


def run_evaluation(
    cfg: Config, device: torch.device, tokenizer: tiktoken.Encoding
) -> None:
    """
    Evaluate the trained GPT-2 language classifier on the test dataset.

    This function prepares the test data loader, loads the pre-trained model 
    weights from the disk, and calculates the overall accuracy of the model.

    Parameters
    ----------
    cfg : Config
        The main configuration object containing meta paths and training settings.
    device : torch.device
        The device (CPU or CUDA) on which to perform computation.
    tokenizer : tiktoken.Encoding
        The BPE tokenizer used to encode the text dataset.

    Returns
    -------
    None
    """
    print("Preparing test dataset for evaluation...")
    test_dataset = LanguageDataset(
        "test", tokenizer, max_length=cfg.model.max_length, cache_dir=cfg.meta.data_dir
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.training.batch_size, shuffle=False
    )

    print(f"Loading trained weights from {cfg.meta.weights_path}...")
    try:
        model = setup_model(cfg, device, load_weights=True)
    except FileNotFoundError:
        print(f"Error: {cfg.meta.weights_path} not found. Run training first.")
        return

    print("\nStarting Standalone Evaluation on Test Set...")
    test_acc = calc_accuracy_loader(test_loader, model, device)
    print(f"--- Final Test Accuracy: {test_acc * 100:.2f}% ---")
