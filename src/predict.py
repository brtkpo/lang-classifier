import torch
from .dataset import ID2LABEL
from src.model import setup_model


def predict_language(
    text, model, tokenizer, device, max_length=128, pad_token_id=50256
):
    model.eval()

    input_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    input_ids = input_ids[:max_length]
    input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]

    predicted_label_id = torch.argmax(logits, dim=-1).item()
    return ID2LABEL[predicted_label_id]


def run_inference(cfg, device, tokenizer):
    try:
        model = setup_model(cfg, device, load_weights=True)
    except FileNotFoundError:
        print(f"Error: {cfg.meta.weights_path} not found. Run training first.")
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
