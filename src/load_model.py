from pathlib import Path
from shutil import copy
from huggingface_hub import hf_hub_download
from .config import Config

def load_model(cfg: Config) -> Path:
    """
    Returns the path to the model checkpoint.
    Downloads from HF if not present locally.

    Parameters
    ----------
    cfg : Config

    Returns
    -------
    Path
        Path to the local checkpoint file.
    """
    meta = cfg.meta
    local_path = Path(meta.weights_path)
    
    local_path.parent.mkdir(exist_ok=True, parents=True)

    if local_path.exists():
        return local_path
    else:
        print("Checkpoint not found locally. Downloading from Hugging Face...")
        hf_path = hf_hub_download(
            repo_id="brtkpo/lang-classifier", 
            filename=local_path.name
        )
        copy(hf_path, local_path)
        print(f"Downloaded and saved to {local_path}")
        return local_path