# utils/preprocessing.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import math
from collections import Counter

# Global variables
accepted_chars = '0123456789abcdefghijklmnopqrstuvwxyz!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
splits_chars = ['://', '//', '/', '.', '_', '=', '-']
n_letters = len(accepted_chars)

class URLDataset(Dataset):
    def __init__(self, urls, labels, char_to_idx, max_len):
        self.urls = urls
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        url = self.urls[idx]
        label = self.labels[idx]
        url_encoded = [self.char_to_idx.get(c, 0) for c in url]
        if len(url_encoded) < self.max_len:
            url_encoded += [0] * (self.max_len - len(url_encoded))
        else:
            url_encoded = url_encoded[:self.max_len]
        return torch.tensor(url_encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def build_char_to_idx():
    return {char: idx + 1 for idx, char in enumerate(accepted_chars)}

def preprocess_urls(urls, labels):
    urls = [url.lower() for url in urls]
    filtered_urls = []
    filtered_labels = []
    for url, label in zip(urls, labels):
        if all(c in accepted_chars for c in url):
            filtered_urls.append(url)
            filtered_labels.append(label)
    return filtered_urls, filtered_labels

def load_data(file_path, url_num=10000, num_classes=2):
    df = pd.read_csv(file_path)
    urls = df.iloc[:, 0].values
    labels = df.iloc[:, 1].values

    urls, labels = preprocess_urls(urls, labels)

    if num_classes == 2:
        labels = [1 if label != 'benign' else 0 for label in labels]
    else:
        label_mapping = {"benign": 0, "malware": 1, "phishing": 2, "defacement": 3}
        labels = [label_mapping.get(label, 4) for label in labels]

    urls = np.array(urls)
    labels = np.array(labels)

    # Sample data
    if url_num < len(urls):
        indices = np.random.permutation(len(urls))[:url_num]
        urls = urls[indices]
        labels = labels[indices]

    return urls, labels

def build_dataloaders(file_path, url_num=10000, batch_size=32, max_len=200, num_classes=2):
    urls, labels = load_data(file_path, url_num, num_classes)
    char_to_idx = build_char_to_idx()

    X_train, X_temp, y_train, y_temp = train_test_split(urls, labels, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_dataset = URLDataset(X_train, y_train, char_to_idx, max_len)
    val_dataset = URLDataset(X_val, y_val, char_to_idx, max_len)
    test_dataset = URLDataset(X_test, y_test, char_to_idx, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(char_to_idx) + 1
