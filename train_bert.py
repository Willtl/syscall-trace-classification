import argparse
import pickle
import random
import warnings

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, AutoModelForSequenceClassification

from utils import ExperimentTracker, set_random_seeds, build_custom_tokenizer, custom_collate_fn, pool_logits


# 0.9734 with 16 batch size and 512 max len with 15% masking (BERT)
# 0.9733 with 64 batch size and 256 max len with 15% masking (BERT)
# 0.9744 with 64 batch size and 256 max len with 15% masking (huawei-noah/TinyBERT_General_4L_312D)
# 0.9763 with 16 batch size and 512 max len with 15% masking (huawei-noah/TinyBERT_General_4L_312D)
# 0.9836 with 64 batch size and 256 max len with 15% masking (deepset/tinyroberta-squad2)
# 0.9794 with 16 batch size and 512 max len with 15% masking (deepset/tinyroberta-squad2)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a BERT model on syscall dataset")

    # Dataset and Experiment Configuration
    parser.add_argument("--dataset_path", type=str, default='datasets/ADFA-LD/processed_sequences.pkl', help="Path to the dataset")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment")

    # Optimization Parameters
    parser.add_argument("--model", type=str, default='deepset/tinyroberta-squad2',
                        choices=['bert-base-uncased', 'distilbert-base-uncased', 'huawei-noah/TinyBERT_General_4L_312D'], help="Type of model to use")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum length of the input sequences (must be between 32 and 512)")

    args = parser.parse_args()
    args.experiment_name = args.model

    # Validate max_len argument
    if not (32 <= args.max_len <= 512):
        warnings.warn("max_len should be between 32 and 512. Clamping to this range.")
        args.max_len = max(32, min(args.max_len, 512))  # Clamp the value within the range

    return args


class SyscallDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_len, training, chunk_overlap=0.1, mask_pct=0.15):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.training = training
        self.chunk_overlap = chunk_overlap
        self.mask_pct = mask_pct

        if training:
            self.tokenizer.enable_truncation(max_length=self.max_len)
            self.tokenizer.enable_padding(length=self.max_len)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        sequence = self.sequences[item]
        label = self.labels[item]

        if self.training:
            start = random.randint(0, max(0, len(sequence) - self.max_len))
            sequence = sequence[start:start + self.max_len]

            sequence = " ".join(map(str, sequence))
            encoding = self.tokenizer.encode(sequence)
            input_ids = torch.tensor(encoding.ids, dtype=torch.long)
            attention_mask = torch.tensor(encoding.attention_mask, dtype=torch.long)

            if self.mask_pct > 0:
                num_tokens_to_mask = int(len(input_ids) * self.mask_pct)
                mask_indices = np.random.choice(len(input_ids), num_tokens_to_mask, replace=False)
                input_ids[mask_indices] = self.tokenizer.token_to_id('[MASK]')

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            # For validation/testing, split the sequence into chunks
            chunks = self.chunk_validation(sequence)
            results = []
            for chunk in chunks:
                encoding = self.tokenizer.encode(chunk)
                results.append({
                    'input_ids': torch.tensor(encoding.ids, dtype=torch.long),
                    'attention_mask': torch.tensor(encoding.attention_mask, dtype=torch.long),
                    'labels': torch.tensor(label, dtype=torch.long),
                    'chunk_index': torch.tensor(item, dtype=torch.long)
                })
            return results

    def chunk_validation(self, sequence):
        overlap = int(self.max_len * self.chunk_overlap)
        start_indexes = list(range(0, len(sequence) - self.max_len + 1, self.max_len - overlap))

        if not start_indexes or start_indexes[-1] + self.max_len < len(sequence):
            start_indexes.append(len(sequence) - self.max_len if len(sequence) > self.max_len else 0)

        chunks = [sequence[i: i + self.max_len] for i in start_indexes]
        chunks = [" ".join(map(str, chunk)) for chunk in chunks]
        return chunks


def load_data(file_path, batch_size, max_len, use_sampler=True):
    # Load data from a pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Extract sequences and labels from data
    sequences = [item['sequence'] for item in data['sequence_data']]
    labels = [item['label'] for item in data['sequence_data']]

    # Split data into training and testing sets before label binarization
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(
        sequences, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Binarize labels
    train_labels = ['normal' if label == 'normal' else 'malware' for label in train_labels]
    test_labels = ['normal' if label == 'normal' else 'malware' for label in test_labels]

    # Encode labels
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.transform(test_labels)

    # Initialize tokenizer
    tokenizer = build_custom_tokenizer(sequences)

    # Create dataset objects
    train_dataset = SyscallDataset(train_sequences, train_labels, tokenizer, max_len=max_len, training=True)
    test_dataset = SyscallDataset(test_sequences, test_labels, tokenizer, max_len=max_len, training=False)

    # Handle sampling for imbalanced dataset
    if use_sampler:
        # Calculate weights for each class
        class_sample_count = np.unique(train_labels, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[train_labels]
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        # Use weighted sampler for the DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    else:
        # Use simple shuffling if no sampler is used
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_loader = DataLoader(test_dataset, collate_fn=custom_collate_fn, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, label_encoder, len(train_dataset), len(test_dataset)


def initialize_model(num_labels, learning_rate, weight_decay, steps_per_epoch, epochs, model_type, epochs_warmup=1):
    # Adjust model creation based on the input model_type
    if model_type == 'bert-base-uncased':
        model = BertForSequenceClassification.from_pretrained(model_type, num_labels=num_labels)
    elif model_type == 'distilbert-base-uncased':
        model = DistilBertForSequenceClassification.from_pretrained(model_type, num_labels=num_labels)
    else:  # Assuming TinyBERT or other compatible models
        model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=num_labels)

    # Reset network weights randomly
    model.init_weights()

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Define lr scheduler
    warmup_steps = steps_per_epoch * epochs_warmup
    total_steps = epochs * steps_per_epoch
    pct_start = warmup_steps / total_steps
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=pct_start,
                           anneal_strategy='cos')

    # Define Criterion
    criterion = torch.nn.CrossEntropyLoss()
    return model, optimizer, scheduler, criterion


def train_epoch(model, data_loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = []
    progress_bar = tqdm(data_loader, desc="Training", leave=True)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss.append(loss.item())

        current_lr = scheduler.get_last_lr()[0]
        mean_loss = np.mean(total_loss)
        progress_bar.set_description(f"Training (LR: {current_lr:.2e}, Loss: {mean_loss})")
    return np.mean(total_loss)


def evaluate_model(model, data_loader, criterion, device, pooling='logits'):
    model.eval()
    total_loss = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            if pooling is not None and 'chunk_index' in batch:
                chunk_indices = batch['chunk_index'].to(device)
                if pooling == 'logits':
                    logits, labels = pool_logits(logits, labels, chunk_indices)

            loss = criterion(logits, labels)
            total_loss.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def main():
    set_random_seeds()

    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, label_encoder, train_size, test_size = load_data(args.dataset_path, args.batch_size, max_len=args.max_len)

    steps_per_epoch = train_size // args.batch_size
    model, optimizer, scheduler, criterion = initialize_model(len(label_encoder.classes_), args.learning_rate, args.weight_decay, steps_per_epoch, args.epochs, args.model)
    model.to(device)

    experiment_tracker = ExperimentTracker(model, optimizer, scheduler, label_encoder, args)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        train_preds, train_labels = evaluate_model(model, train_loader, criterion, device)
        test_preds, test_labels = evaluate_model(model, test_loader, criterion, device)

        current_lr = scheduler.get_last_lr()[0]
        experiment_tracker.update_and_save(epoch, train_loss, train_preds, train_labels, test_preds, test_labels, current_lr)
        print(experiment_tracker)


if __name__ == "__main__":
    main()
