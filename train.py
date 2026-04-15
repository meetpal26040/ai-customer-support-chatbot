import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

with open('intents.json') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = sorted(set([stem(w) for w in all_words if w not in ignore_words]))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __len__(self):
        return len(X_train)

    def __getitem__(self, index):
        return X_train[index], y_train[index]

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

model = NeuralNet(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(500):
    for (words, labels) in train_loader:
        outputs = model(words.float())
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Training complete")

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

torch.save(data, "data.pth")