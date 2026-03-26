import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Any

from .download import download_and_load_gpt2
from .config import Config

class GPTDatasetV1(Dataset):
    """
    A PyTorch Dataset for basic text generation tasks using GPT.

    Parameters
    ----------
    txt : str
        The full text data to be processed.
    tokenizer : tiktoken.Encoding
        The BPE tokenizer used to encode the text.
    max_length : int
        The sequence length (context window) for the model.
    stride : int
        The step size for moving the window across the text.
    """
    def __init__(
        self, 
        txt: str, 
        tokenizer: tiktoken.Encoding, 
        max_length: int, 
        stride: int
    ) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for the GPTDatasetV1.

    Parameters
    ----------
    txt : str
        The full text data.
    batch_size : int, default=4
        Number of samples per batch.
    max_length : int, default=256
        Maximum sequence length.
    stride : int, default=128
        Stride size for the sliding window.
    shuffle : bool, default=True
        Whether to shuffle the dataset.
    drop_last : bool, default=True
        Whether to drop the last incomplete batch.
    num_workers : int, default=0
        Number of subprocesses for data loading.

    Returns
    -------
    DataLoader
        The configured PyTorch DataLoader.
    """
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Attention mechanism.

    Parameters
    ----------
    d_in : int
        Input dimensionality.
    d_out : int
        Output dimensionality.
    context_length : int
        Maximum sequence length (for the causal mask).
    dropout : float
        Dropout probability.
    num_heads : int
        Number of attention heads.
    qkv_bias : bool, default=False
        Whether to use bias in the Q, K, V linear projections.
    """
    def __init__(
        self, 
        d_in: int, 
        d_out: int, 
        context_length: int, 
        dropout: float, 
        num_heads: int, 
        qkv_bias: bool = False
    ) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class LayerNorm(nn.Module):
    """
    Layer Normalization module.

    Parameters
    ----------
    emb_dim : int
        The dimensionality of the embeddings to be normalized.
    """
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network used in the Transformer block.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration dictionary containing 'emb_dim'.
    """
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    A single Transformer Block comprising Multi-Head Attention and a Feed-Forward Network.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration dictionary containing architectural hyperparameters.
    """
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """
    The main GPT-2 Model architecture.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration dictionary specifying the model's dimensions and hyperparameters.
    """
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int) -> torch.Tensor:
    """
    Greedy text generation using the trained model.

    Parameters
    ----------
    model : nn.Module
        The trained language model.
    idx : torch.Tensor
        The current sequence of token IDs.
    max_new_tokens : int
        The number of new tokens to generate.
    context_size : int
        The maximum context size the model can handle.

    Returns
    -------
    torch.Tensor
        The expanded sequence of token IDs including the newly generated tokens.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def assign(left: nn.Parameter | torch.Tensor, right: np.ndarray | torch.Tensor) -> nn.Parameter:
    """
    Helper function to assign pre-trained weights to a PyTorch parameter.

    Parameters
    ----------
    left : nn.Parameter | torch.Tensor
        The target PyTorch parameter.
    right : np.ndarray | torch.Tensor
        The source weights to assign.

    Returns
    -------
    nn.Parameter
        The updated parameter.

    Raises
    ------
    ValueError
        If the shape of the target and source do not match.
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt: GPTModel, params: dict[str, Any]) -> None:
    """
    Load pre-trained OpenAI weights into the GPT model structure.

    Parameters
    ----------
    gpt : GPTModel
        The instantiated GPT model.
    params : dict[str, Any]
        The dictionary containing extracted TensorFlow weights.

    Returns
    -------
    None
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].norm1.weight = assign(
            gpt.trf_blocks[b].norm1.weight, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.bias = assign(
            gpt.trf_blocks[b].norm1.bias, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.weight = assign(
            gpt.trf_blocks[b].norm2.weight, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.bias = assign(
            gpt.trf_blocks[b].norm2.bias, params["blocks"][b]["ln_2"]["b"]
        )

    gpt.final_norm.weight = assign(gpt.final_norm.weight, params["g"])
    gpt.final_norm.bias = assign(gpt.final_norm.bias, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    """
    Encode text into token IDs.

    Parameters
    ----------
    text : str
        The input string.
    tokenizer : tiktoken.Encoding
        The BPE tokenizer.

    Returns
    -------
    torch.Tensor
        A 2D tensor containing the encoded token IDs.
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    """
    Decode token IDs back into text.

    Parameters
    ----------
    token_ids : torch.Tensor
        The tensor containing token IDs.
    tokenizer : tiktoken.Encoding
        The BPE tokenizer.

    Returns
    -------
    str
        The decoded string.
    """
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def setup_model(cfg: Config, device: torch.device, load_weights: bool = False) -> GPTModel:
    """
    Initialize the GPT-2 model, optionally download pre-trained weights,
    and adapt the classification head for fine-tuning.

    Parameters
    ----------
    cfg : Config
        The configuration object with model hyperparameters.
    device : torch.device
        The device (CPU or CUDA) to map the model to.
    load_weights : bool, default=False
        If True, loads fine-tuned local weights. If False, downloads and loads
        original GPT-2 pre-trained weights from the internet.

    Returns
    -------
    GPTModel
        The ready-to-use GPT model.
    """
    model_config_dict = {
        "vocab_size": cfg.model.vocab_size,
        "context_length": cfg.model.context_length,
        "drop_rate": cfg.model.drop_rate,
        "qkv_bias": cfg.model.qkv_bias,
        "emb_dim": cfg.model.emb_dim,
        "n_layers": cfg.model.n_layers,
        "n_heads": cfg.model.n_heads,
    }

    model = GPTModel(model_config_dict)

    if load_weights:
        model.out_head = torch.nn.Linear(
            in_features=cfg.model.emb_dim, out_features=cfg.model.num_classes
        )
        model.load_state_dict(
            torch.load(cfg.meta.weights_path, map_location=device, weights_only=True)
        )
    else:
        print(f"Downloading GPT-2 ({cfg.model.model_size}) weights...")
        settings, params = download_and_load_gpt2(
            model_size=cfg.model.model_size, models_dir="gpt2"
        )
        load_weights_into_gpt(model, params)

        for param in model.parameters():
            param.requires_grad = False

        model.out_head = torch.nn.Linear(
            in_features=cfg.model.emb_dim, out_features=cfg.model.num_classes
        )

        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True

    model.to(device)
    return model
