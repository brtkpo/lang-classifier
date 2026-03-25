import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MetaConfig:
    mode: str = "train"
    weights_path: str = "language_classifier.pth"
    data_dir: str = "data"


@dataclass
class ModelConfig:
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
    batch_size: int = 64
    epochs: int = 1
    lr: float = 5e-5
    weight_decay: float = 0.1


@dataclass
class Config:
    meta: MetaConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: Path) -> Config:
    with open(path) as f:
        raw = json.load(f)

    meta = MetaConfig(**raw.get("meta", {}))
    model = ModelConfig(**raw.get("model", {}))
    training = TrainingConfig(**raw.get("training", {}))

    return Config(meta=meta, model=model, training=training)
