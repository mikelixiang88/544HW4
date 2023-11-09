import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR  # Import the scheduler
from torch.nn.utils.rnn import pad_sequence
import datasets
from conlleval import evaluate
import itertools
from collections import Counter


# Build the vocabularies
def build_vocabularies(dataset):
    # Build word vocabulary
    word_counter = Counter()
    for sample in dataset['train']:
        word_counter.update(sample['tokens'])
    word_vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counter.items()]

    word2idx = {word: idx for idx, word in enumerate(word_vocab)}

    # Build label vocabulary
    label_vocab = dataset['train'].features['labels'].feature.names
    label2idx = {label: idx for idx, label in enumerate(label_vocab)}

    return word2idx, label2idx


dataset = datasets.load_dataset("conll2003")
# Remove 'pos tags' and 'chunk tags' columns
dataset = dataset.remove_columns(['pos_tags', 'chunk_tags'])
# Rename 'ner tags' to 'labels'
dataset = dataset.rename_column('ner_tags', 'labels')
word2idx, label2idx = build_vocabularies(dataset)

# Define a Dataset class
class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.word2idx = word2idx
        self.label2idx = label2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_ids = torch.tensor([self.word2idx.get(token, self.word2idx['<UNK>']) for token in sample['tokens']], dtype=torch.long)
        labels = torch.tensor(sample['labels'], dtype=torch.long)
        return input_ids, labels

# Function to convert words to IDs and labels to IDs
def convert_tokens_labels_to_ids(dataset, word2idx, label2idx):
    def convert_word_to_id(sample):
        return {
            'tokens': [word2idx.get(token, word2idx['<UNK>']) for token in sample['tokens']],
            'labels': [label2idx[label] for label in sample['labels']]
        }
    return dataset.map(convert_word_to_id)

# Padding function
def pad_collate(batch):
    # Unzip the batch
    input_ids, labels = zip(*batch)
    # Pad the sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=word2idx['<PAD>'])
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=label2idx['O'])
    return input_ids_padded, labels_padded

# Check if CUDA is available and set the default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the BiLSTM model
class BiLSTMForNER(nn.Module):
    def __init__(self, vocab_size, label_vocab_size, embedding_dim, lstm_hidden_dim, linear_output_dim, num_lstm_layers, lstm_dropout):
        super(BiLSTMForNER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=num_lstm_layers, 
                              dropout=lstm_dropout, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(lstm_hidden_dim * 2, linear_output_dim)  # Times 2 because of bidirection
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_output_dim, label_vocab_size)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.bilstm(embedded)
        linear_out = self.linear(lstm_out)
        elu_out = self.elu(linear_out)
        logits = self.classifier(elu_out)
        return logits

# Create DataLoader
train_dataset = NERDataset(dataset['train'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
# Create DataLoaders for validation and test data
val_dataset = NERDataset(dataset['validation'])
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=pad_collate)

test_dataset = NERDataset(dataset['test'])
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=pad_collate)

vocab_size = len(word2idx)
label_vocab_size=9
model = BiLSTMForNER(vocab_size, 9, 100, 256, 128, 1, 0.33)
model = model.to(device)
# Map the labels back to their corresponding tag strings
idx2tag = {idx: tag for tag, idx in label2idx.items()}

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>']).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Define the learning rate scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)


def evaluate_model(model, data_loader):
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for input_ids, labels in data_loader:
            # Forward pass
            input_ids.to(device)
            outputs = model(input_ids)

            # Convert to probabilities and get the predicted labels
            preds = outputs.argmax(-1)

            # Iterate over the batch and collect non-padded values
            for i in range(len(labels)):
                true_sequence = labels[i][labels[i] != word2idx['<PAD>']]
                pred_sequence = preds[i][:len(true_sequence)]  # Match true sequence length

                true_labels.extend(true_sequence.tolist())
                pred_labels.extend(pred_sequence.tolist())

    # Convert to tags
    true_tags = [idx2tag[idx] for idx in true_labels]
    pred_tags = [idx2tag[idx] for idx in pred_labels]

    # Calculate metrics
    precision, recall, f1 = evaluate(true_tags, pred_tags)
    return precision, recall, f1

# Training loop (simplified, assuming all other necessary components are defined)
for epoch in range(20):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, label_vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
    scheduler.step()  # Update the learning rate
    # Evaluate on validation set
    val_precision, val_recall, val_f1 = evaluate_model(model, val_loader)
    print(f"Validation - Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}")
    
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


# Evaluate on test set
precision, recall, f1 = evaluate_model(model, test_loader)
print(f"Test - Precision: {precision}, Recall: {recall}, F1: {f1}")
