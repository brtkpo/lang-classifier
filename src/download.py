import os
import urllib.request
from typing import Any

import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size: str, models_dir: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Download GPT-2 model weights and settings from OpenAI and load them into memory.

    Parameters
    ----------
    model_size : str
        The size of the GPT-2 model to download (e.g., "124M", "355M").
    models_dir : str
        The directory where the downloaded model files will be saved.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        A tuple containing:
        - settings: A dictionary containing the model's hyperparameters.
        - params: A nested dictionary containing the loaded TensorFlow weights.

    Raises
    ------
    ValueError
        If the specified `model_size` is not in the list of allowed sizes.
    """
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]

    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(
        open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8")
    )
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url: str, destination: str, backup_url: str | None = None) -> None:
    """
    Download a file from a URL to a specified destination with a progress bar.

    If the primary URL fails, attempts to download from a backup URL if provided.
    Skips downloading if the file already exists and matches the remote file size.

    Parameters
    ----------
    url : str
        The primary URL of the file to download.
    destination : str
        The local file path where the downloaded file should be saved.
    backup_url : str | None, default=None
        An optional fallback URL to use if the primary URL fails.

    Returns
    -------
    None
    """
    def _attempt_download(download_url):
        with urllib.request.urlopen(download_url) as response:
            file_size = int(response.headers.get("Content-Length", 0))

            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return True

            block_size = 1024

            progress_bar_description = os.path.basename(download_url)
            with tqdm(
                total=file_size,
                unit="iB",
                unit_scale=True,
                desc=progress_bar_description,
            ) as progress_bar:
                with open(destination, "wb") as file:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            return True

    try:
        if _attempt_download(url):
            return
    except (urllib.error.HTTPError, urllib.error.URLError):
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except urllib.error.HTTPError:
                pass

        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_gpt2_params_from_tf_ckpt(ckpt_path: str, settings: dict[str, Any]) -> dict[str, Any]:
    """
    Parse TensorFlow checkpoint files and extract weights into a structured dictionary.

    Parameters
    ----------
    ckpt_path : str
        Path to the TensorFlow checkpoint file.
    settings : dict[str, Any]
        Dictionary containing model hyperparameters (needs "n_layer").

    Returns
    -------
    dict[str, Any]
        A nested dictionary mapping component names to NumPy arrays containing
        the model weights.
    """
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        variable_name_parts = name.split("/")[1:]

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
