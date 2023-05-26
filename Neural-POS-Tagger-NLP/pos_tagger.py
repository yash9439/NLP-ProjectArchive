import re
import numpy as np 
import pandas as pd 
import conllu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt



# Loading dataset
import torchvision
from torch.utils.data import Dataset
import numpy as np
import math


class TreeBankDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = f.read()
        parsed_data = conllu.parse(data)
        self.data = []
        for sentence in parsed_data:
            tagged_sentence_word = []
            tagged_sentence_tag = []
            for token in sentence:
                word = token["form"]
                pos = token["upos"]
                tagged_sentence_word.append(word)
                tagged_sentence_tag.append(pos)
            self.data.append((tagged_sentence_word,tagged_sentence_tag))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    


dataset_train = TreeBankDataset("/Dataset/en_atis-ud-train.conllu")
dataset_test = TreeBankDataset("/Dataset/en_atis-ud-test.conllu")
dataset_dev = TreeBankDataset("/Dataset/en_atis-ud-dev.conllu")


# training sentences and their corresponding word-tags
training_data = dataset_train


# create a dictionary that maps words to indices
word2idx = {}
tag2idx = {}
idx2tag = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

for sent, tags in training_data:
    for tag in tags:
        if tag not in tag2idx:
            tag2idx[tag] = len(tag2idx)
            idx2tag[len(idx2tag)] = tag

word2idx["UNK"] = len(word2idx)
tag2idx['UNK'] = len(tag2idx)
idx2tag[len(idx2tag)] = 'UNK'



import numpy as np

def texts_to_sequences(text, word_index):
    sequence = []
    for token in text:
        if token in word_index.keys():
            sequence.append(word_index[token])
        else:
            sequence.append(word_index['UNK'])
    sequence = np.array(sequence)
    return torch.from_numpy(sequence)







# Model

class BiLSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, dropout = 0.3):
        ''' Initialize the layers of this model.'''
        super(BiLSTMTagger, self).__init__()
        
        self.hidden_dim = hidden_dim

        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.word_embeddings.weight.data.copy_(pretrained_embeddings)
        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        
        self.dropout = nn.Dropout(dropout)
        
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        
        # initialize the hidden state (see code below)
        self.hidden = self.init_hidden()

        
    def init_hidden(self):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim))

    def forward(self, sentence):
        ''' Define the feedforward behavior of the model.'''
        # create embedded word vectors for each word in a sentence
        embeds = self.word_embeddings(sentence)
        
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        
        lstm_out = self.dropout(lstm_out)
        
        # get the scores for the most likely tag for a word
        tag_outputs = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_outputs, dim=1)
        
        return tag_scores
    



# Loading Model

EMBEDDING_DIM = 300
HIDDEN_DIM = 128

# instantiate our model
model = BiLSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2idx), len(tag2idx))

# define our loss and optimizer
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())


PATH = './savedModel.pth'
model.load_state_dict(torch.load(PATH))





test_sentence = input("Enter a Sentence: ")
test_sentence = re.sub(r'[^\w\s]', '', test_sentence)
test_sentence = test_sentence.lower().split()
# test_sentence = "i would like the cheapest flight from pittsburgh to atlanta leaving april twenty fifth and returning may sixth".lower().split()


inputs = texts_to_sequences(test_sentence, word2idx)
inputs = inputs
tag_scores = model(inputs)
_, predicted_tags = torch.max(tag_scores, 1)


ct = 0
for i in predicted_tags:
    print(test_sentence[ct], end="\t")
    print(idx2tag[i.item()])
    ct += 1
