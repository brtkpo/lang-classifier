# Language Classifier

Deep learning model for text language identification based on the GPT-2 architecture. 
The pipeline automatically downloads pre-trained GPT-2 weights from OpenAI, adapts the final classification head, and fine-tunes the network to predict the language of an input text out of 20 supported languages.

---

## Dataset

This project uses the **Language Identification dataset** from Hugging Face:
https://huggingface.co/datasets/papluca/language-identification

---

## Installation

This project uses **uv** for Python dependency management.

### 1. Install uv

#### Windows (PowerShell)
```bash
pip install uv
uv sync
```
Run the project using:
```bash
uv run main.py -c config.json
```
---
## Configuration   
The project is controlled via a JSON configuration file passed with:

```bash
uv run main.py -c config.json
```
#### JSON Structure
```json
{
  "meta": {
    "mode": "predict",
    "weights_path": "language_classifier.pth",
    "data_dir": "data"
  },
  "model": {
    "model_size": "124M",
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": true,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "num_classes": 20,
    "max_length": 128
  },
  "training": {
    "batch_size": 64,
    "epochs": 1,
    "lr": 0.00005,
    "weight_decay": 0.1
  }
}
```

#### meta
Controls the overall pipeline behavior.  
- **mode** – what the program should do: `"train"` (download pre-trained OpenAI weights, fine-tune the model, and evaluate), `"evaluate"` (run standalone evaluation on the test set), or `predict` (launch an interactive CLI for testing custom sentences)  
- **weights_path** – path to save/load the fine-tuned model weights  
- **data_dir** – folder where results (CSV, images) are saved  
- **checkpoint_dir** – folder where the Hugging Face dataset is cached

#### model
Controls the GPT-2 architecture hyperparameters.
- **model_size** – original OpenAI weights to download if training from scratch
- **max_length** – maximum sequence length for token truncation/padding 

#### training
Controls the training process.

---

## Model Weights

The trained model weights (`language_classifier.pth`) are not included in the repository.

At runtime:
- the model is loaded from `language_classifier.pth` if available,
- otherwise it is automatically downloaded from Hugging Face:
https://huggingface.co/brtkpo/lang-classifier

---

## Example Usage Of Prediction

The model supports detecting text in **20 languages**:
- Arabic (ar), Bulgarian (bg), German (de), Modern Greek (el), English (en)
- Spanish (es), French (fr), Hindi (hi), Italian (it), Japanese (ja)
- Dutch (nl), Polish (pl), Portuguese (pt), Russian (ru), Swahili (sw)
- Thai (th), Turkish (tr), Urdu (ur), Vietnamese (vi), Chinese (zh)

### Running Prediction Mode

1. **Set the mode to `predict` in `config.json`:**
   ```json
   {
     "meta": {
       "mode": "predict",
       "weights_path": "language_classifier.pth",
       "data_dir": "data"
     }
   }
   ```

2. **Run the program:**
   ```bash
   uv run main.py -c config.json
   ```

3. **Enter text to classify:**
   The program will start an interactive CLI where you can type or paste text.
   The model will predict the language and display the result.

4. **Exit the program:**
   - Type `/exit` to exit the interactive prediction mode