import datasets
from transformers import BertTokenizerFast
from conlleval import evaluate
import itertools


# Define a Dataset class
class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.data[idx]['input_ids'], dtype=torch.long),
            'labels': torch.tensor(self.data[idx]['labels'], dtype=torch.long)
        }
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


# Convert tokens to IDs
def convert_word_to_id(sample):
    return {
        'input_ids': [word2idx.get(token, word2idx['UNK']) for token in sample['tokens']],
        'labels': sample['labels']
    }


dataset = datasets.load_dataset("conll2003")
word2idx, label2idx = build_vocabularies(dataset)

# Remove 'pos tags' and 'chunk tags' columns
dataset = dataset.remove_columns(['pos_tags', 'chunk_tags'])

# Rename 'ner tags' to 'labels'
dataset = dataset.rename_column('ner_tags', 'labels')
dataset.map(convert_word_to_id)
# Create DataLoader
train_dataset = NERDataset(dataset['train'])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define the BiLSTM model
class BiLSTMForNER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, linear_output_dim, num_lstm_layers, lstm_dropout):
        super(BiLSTMForNER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=num_lstm_layers, 
                              dropout=lstm_dropout, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(lstm_hidden_dim * 2, linear_output_dim)  # Times 2 because of bidirection
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_output_dim, len(label2idx))  # Assuming label2idx is defined

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.bilstm(embedded)
        linear_out = self.linear(lstm_out)
        elu_out = self.elu(linear_out)
        logits = self.classifier(elu_out)
        return logits

# Assuming you have a vocab_size and a label2idx dictionary
vocab_size = len(word2idx)
model = BiLSTMForNER(vocab_size, 100, 256, 128, 1, 0.33)

# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):  # Let's assume 5 epochs
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids']
        labels = batch['labels']

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids)
        
        # Reshape outputs and labels to compute loss
        outputs = outputs.view(-1, outputs.shape[-1])
        labels = labels.view(-1)

        # Compute loss and backpropagate
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Map the labels back to their corresponding tag strings
idx2tag = {idx: tag for tag, idx in label2idx.items()}

# Implement the prediction loop for the validation set
model.eval()  # Set the model to evaluation mode
predictions = []
with torch.no_grad():
    for batch in DataLoader(NERDataset(dataset['validation']), batch_size=16):
        input_ids = batch['input_ids']
        # Forward pass
        outputs = model(input_ids)
        # Get the index of the highest logit
        predictions.extend(outputs.argmax(dim=-1).tolist())

# Convert predictions and labels to tag names
pred_tags = [
    [idx2tag.get(idx) for idx in sentence]
    for sentence in predictions
]
true_tags = [
    [idx2tag.get(idx) for idx in sentence['labels']]
    for sentence in dataset['validation']
]

# Flatten the lists for evaluation
flattened_preds = [tag for sentence in pred_tags for tag in sentence]
flattened_trues = [tag for sentence in true_tags for tag in sentence]

# Evaluate the predictions
precision, recall, f1 = evaluate(flattened_trues, flattened_preds)

print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")


