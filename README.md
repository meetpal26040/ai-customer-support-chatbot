# AI Customer Support Chatbot

## Overview
This project is an NLP-based chatbot that classifies user queries into predefined intents using a neural network model.

## Features
- Intent classification using PyTorch
- Text preprocessing with tokenization and stemming
- Bag-of-words feature extraction
- Confidence-based response system

## Tech Stack
- Python
- PyTorch
- NLTK

## How it Works
1. User input is tokenized
2. Words are converted into a bag-of-words vector
3. The vector is passed into a neural network
4. The model predicts the most likely intent
5. A response is returned based on confidence score

## How to Run

### 1. Train the model
python train.py


### 2. Run chatbot
python chat.py


## Example

You: Where is my order?
Bot: You can track your order using your order ID.
