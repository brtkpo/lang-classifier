import torch
import tiktoken

from .dataset import ID2LABEL
from src.model import setup_model
from src.config import Config
from .load_model import load_model


def predict_language(
    text: str, 
    model: torch.nn.Module, 
    tokenizer: tiktoken.Encoding, 
    device: torch.device, 
    max_length: int = 128, 
    pad_token_id: int = 50256
) -> str:
    """
    Predict the language of a given text using the trained GPT-2 model.

    Parameters
    ----------
    text : str
        The input text to classify.
    model : torch.nn.Module
        The trained GPT-2 classification model.
    tokenizer : tiktoken.Encoding
        The BPE tokenizer used to encode the input text.
    device : torch.device
        The device (CPU or CUDA) on which to perform computation.
    max_length : int, default=128
        The maximum sequence length. Inputs longer than this will be truncated.
    pad_token_id : int, default=50256
        The token ID used for padding sequences to `max_length`.

    Returns
    -------
    str
        The predicted language label.
    """
    model.eval()

    input_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    input_ids = input_ids[:max_length]
    input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]

    predicted_label_id = torch.argmax(logits, dim=-1).item()
    return ID2LABEL[predicted_label_id]


def run_inference(
    cfg: Config, device: torch.device, tokenizer: tiktoken.Encoding
) -> None:
    try:
        _ = load_model(cfg)
        
        model = setup_model(cfg, device, load_weights=True)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("Please check Hugging Face repository or run training first.")
        return

    print("\n--- Language Identification CLI (type '/exit' to quit) ---")
    while True:
        text = input("Enter text: ")
        if text.lower() == "/exit":
            break

        lang = predict_language(
            text, model, tokenizer, device, max_length=cfg.model.max_length
        )
        print(f"Identified language: {lang.upper()}\n")
