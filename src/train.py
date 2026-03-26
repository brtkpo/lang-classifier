import torch
import time
import tiktoken
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from .dataset import LanguageDataset
from .model import setup_model
from .config import Config


def calc_accuracy_loader(
    data_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """
    Calculate the accuracy of the model over a data loader.

    Parameters
    ----------
    data_loader : DataLoader
        The PyTorch DataLoader providing input and target batches.
    model : torch.nn.Module
        The neural network model being evaluated.
    device : torch.device
        The device (CPU or CUDA) to perform the calculations on.
    num_batches : int | None, default=None
        Maximum number of batches to evaluate. If None, evaluates the whole loader.

    Returns
    -------
    float
        The accuracy as a fraction (between 0.0 and 1.0).
    """
    model.eval()
    correct_predictions, num_examples = 0, 0
    num_batches = (
        len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    )

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Calculate the Cross-Entropy loss for a single batch.

    Parameters
    ----------
    input_batch : torch.Tensor
        Batch of token IDs.
    target_batch : torch.Tensor
        Batch of target language labels.
    model : torch.nn.Module
        The neural network model.
    device : torch.device
        The device (CPU or CUDA).

    Returns
    -------
    torch.Tensor
        A scalar tensor containing the computed loss.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return torch.nn.functional.cross_entropy(logits, target_batch)


def train_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
) -> None:
    """
    Train the model for a specified number of epochs.

    Parameters
    ----------
    model : torch.nn.Module
        The language classification model to be trained.
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset (used for end-of-epoch checks).
    optimizer : torch.optim.Optimizer
        The optimizer used to update model weights.
    device : torch.device
        The device (CPU or CUDA) for training.
    num_epochs : int
        The total number of training epochs.

    Returns
    -------
    None
    """
    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch [{epoch + 1}/{num_epochs}]",
            leave=True,
        )
        for i, (input_batch, target_batch) in enumerate(loop):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                loop.set_postfix({"Loss": f"{loss.item():.4f}"})

        train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=20)
        val_acc = calc_accuracy_loader(val_loader, model, device, num_batches=20)
        print(f"--- End of Epoch {epoch + 1} ---")
        print(f"Train Acc: {train_acc * 100:.2f}% | Val Acc: {val_acc * 100:.2f}%\n")


def run_training(cfg: Config, device: torch.device, tokenizer: tiktoken.Encoding) -> None:
    """
    Initialize datasets and orchestrate the full training process.

    This function sets up the data loaders for training and validation,
    initializes the model and optimizer, runs the training loop, and saves
    the resulting model weights to the disk.

    Parameters
    ----------
    cfg : Config
        The main configuration object containing model and training settings.
    device : torch.device
        The device (CPU or CUDA) to train on.
    tokenizer : tiktoken.Encoding
        The BPE tokenizer used to encode the datasets.

    Returns
    -------
    None
    """
    print("Preparing datasets...")
    train_dataset = LanguageDataset(
        "train", tokenizer, max_length=cfg.model.max_length, cache_dir=cfg.meta.data_dir
    )
    val_dataset = LanguageDataset(
        "validation",
        tokenizer,
        max_length=cfg.model.max_length,
        cache_dir=cfg.meta.data_dir,
    )
    test_dataset = LanguageDataset(
        "test", tokenizer, max_length=cfg.model.max_length, cache_dir=cfg.meta.data_dir
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.training.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.training.batch_size, shuffle=False
    )

    model = setup_model(cfg, device, load_weights=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    print("Starting training...")
    start_time = time.time()
    train_classifier(
        model, train_loader, val_loader, optimizer, device, cfg.training.epochs
    )
    print(f"Training finished in {(time.time() - start_time) / 60:.2f} minutes.")

    print("\nEvaluating on Test Set...")
    test_acc = calc_accuracy_loader(test_loader, model, device)
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

    torch.save(model.state_dict(), cfg.meta.weights_path)
    print(f"Model saved as {cfg.meta.weights_path}")
