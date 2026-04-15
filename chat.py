import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cpu')

with open('intents.json') as f:
    intents = json.load(f)

data = torch.load("data.pth")

model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"])
model.load_state_dict(data["model_state"])
model.eval()

all_words = data['all_words']
tags = data['tags']

bot_name = "SupportBot"

print("Start chatting (type 'quit' to exit)")

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X.reshape(1, -1)).float()

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    print(f"(Debug) Intent: {tag}, Confidence: {prob.item():.2f}")

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I'm not fully sure. Did you mean something about '{tag}'?")