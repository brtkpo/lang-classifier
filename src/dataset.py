import torch
import tiktoken
from torch.utils.data import Dataset
from datasets import load_dataset

LABELS = [
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "it",
    "ja",
    "nl",
    "pl",
    "pt",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


class LanguageDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing the 'papluca/language-identification' dataset.

    Parameters
    ----------
    split : str
        The dataset split to load (e.g., "train", "validation", "test").
    tokenizer : tiktoken.Encoding
        The tokenizer used to encode the text samples into token IDs.
    max_length : int, default=128
        The maximum sequence length. Sequences longer than this will be truncated,
        and sequences shorter will be padded.
    pad_token_id : int, default=50256
        The token ID used to pad sequences to `max_length`.
    cache_dir : str, default="data"
        The directory where the Hugging Face dataset will be cached.
    """
    
    def __init__(
        self,
        split: str,
        tokenizer: tiktoken.Encoding,
        max_length: int = 128,
        pad_token_id: int = 50256,
        cache_dir: str = "data",
    ) -> None:
        dataset = load_dataset("papluca/language-identification", cache_dir=cache_dir)
        self.data = dataset[split]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset split.

        Returns
        -------
        int
            The total number of text samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label_str = item["labels"]

        label_id = LABEL2ID[label_str]

        encoded = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        encoded = encoded[: self.max_length]

        encoded += [self.pad_token_id] * (self.max_length - len(encoded))

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long),
        )
