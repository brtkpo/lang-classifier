import time
import torch
import tiktoken
from torch.utils.data import DataLoader
from .model import GPTModel, load_weights_into_gpt
from .download import download_and_load_gpt2
from .dataset import LanguageDataset, LABELS


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return torch.nn.functional.cross_entropy(logits, target_batch)

def train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for i, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f}")

        train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=20)
        val_acc = calc_accuracy_loader(val_loader, model, device, num_batches=20)
        print(f"--- End of Epoch {epoch+1} ---")
        print(f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%\n")
        