import torch
from torch.utils.data import Dataset
from datasets import load_dataset

LABELS = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'it', 'ja', 
          'nl', 'pl', 'pt', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

class LanguageDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=128, pad_token_id=50256):
        dataset = load_dataset("papluca/language-identification")
        self.data = dataset[split]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label_str = item["labels"]
        
        label_id = LABEL2ID[label_str]

        encoded = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        
        encoded = encoded[:self.max_length]
        
        encoded += [self.pad_token_id] * (self.max_length - len(encoded))

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long)
        )