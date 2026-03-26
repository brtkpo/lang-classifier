import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MetaConfig:
    """
    General configuration related to experiment setup and file paths.

    Parameters
    ----------
    mode : str, default="train"
        Execution mode of the pipeline (e.g., "train", "evaluate", "predict").
    weights_path : str, default="language_classifier.pth"
        Path where the fine-tuned model weights will be saved or loaded from.
    data_dir : str, default="data"
        Directory where Hugging Face datasets will be cached.
    """
    mode: str = "train"
    weights_path: str = "language_classifier.pth"
    data_dir: str = "data"


@dataclass
class ModelConfig:
    """
    Configuration of the GPT-2 model architecture.

    Parameters
    ----------
    model_size : str, default="124M"
        Size of the pre-trained GPT-2 model to load ("124M", "355M", "774M", "1558M").
    vocab_size : int, default=50257
        Size of the tokenizer's vocabulary (GPT-2 default is 50257).
    context_length : int, default=1024
        Maximum context size (sequence length) the model can handle.
    drop_rate : float, default=0.0
        Dropout probability for regularization.
    qkv_bias : bool, default=True
        Whether to include bias in Query, Key, and Value linear layers.
    emb_dim : int, default=768
        Dimensionality of the token embeddings.
    n_layers : int, default=12
        Number of transformer blocks in the model.
    n_heads : int, default=12
        Number of attention heads in the multi-head attention mechanism.
    num_classes : int, default=20
        Number of output classes for the classification head (e.g., number of languages).
    max_length : int, default=128
        Maximum sequence length to which inputs will be truncated/padded during training.
    """
    model_size: str = "124M"
    vocab_size: int = 50257
    context_length: int = 1024
    drop_rate: float = 0.0
    qkv_bias: bool = True
    emb_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    num_classes: int = 20
    max_length: int = 128


@dataclass
class TrainingConfig:
    """
    Configuration for training hyperparameters.

    Parameters
    ----------
    batch_size : int, default=64
        Number of samples per training/evaluation batch.
    epochs : int, default=1
        Total number of training epochs.
    lr : float, default=5e-5
        Learning rate for the optimizer.
    weight_decay : float, default=0.1
        Weight decay (L2 penalty) factor for the optimizer.
    """
    batch_size: int = 64
    epochs: int = 1
    lr: float = 5e-5
    weight_decay: float = 0.1


@dataclass
class Config:
    """
    Main configuration object grouping all sub-configurations.

    Parameters
    ----------
    meta : MetaConfig
        General experiment and path configuration.
    model : ModelConfig
        Model architecture settings.
    training : TrainingConfig
        Training hyperparameters.
    """
    meta: MetaConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: Path) -> Config:
    """
    Load configuration from a JSON file and convert it into Config dataclasses.

    Parameters
    ----------
    path : Path
        Path to the JSON configuration file.

    Returns
    -------
    Config
        Configuration object containing:
        - meta : MetaConfig with general paths and execution mode
        - model : ModelConfig with architecture settings
        - training : TrainingConfig with training hyperparameters
    """
    with open(path) as f:
        raw = json.load(f)

    meta = MetaConfig(**raw.get("meta", {}))
    model = ModelConfig(**raw.get("model", {}))
    training = TrainingConfig(**raw.get("training", {}))

    return Config(meta=meta, model=model, training=training)
