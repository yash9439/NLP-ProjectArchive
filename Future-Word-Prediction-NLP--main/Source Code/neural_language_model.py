from collections import defaultdict
import sys
import itertools
import time
import math
import torch
from torch import nn
import numpy as np

def get_vocab(train_file):
    freqs = defaultdict(int)
    maxlength = 0
    vocab = []
    prefix_seqs = []

    with open(train_file, "r") as f:
        for line in f:
            tokens = line.split(" ")
            if(len(tokens) > 50):
              continue
            vocab += tokens
            maxlength = max(maxlength, len(tokens))

            for token in tokens:
                freqs[token] += 1

            try:
                local_prefix_seqs = [tokens[0]]
                for token in tokens[1:]:
                    local_prefix_seqs.append(token)
                    prefix_seqs.append(local_prefix_seqs.copy())
            except:
                continue

    vocab = [token for token, freq in freqs.items() if freq >= 5]
    vocab.extend(['<unk>', '<pad>'])
    print(len(vocab))

    vocab_size = len(vocab)
    word_to_index = {word: i for i, word in enumerate(vocab)}
    word_to_index['<unk>'] = vocab_size - 2
    word_to_index['<pad>'] = vocab_size - 1

    prefix_seqs_indexed = [[word_to_index.get(word, word_to_index['<unk>']) for word in seq] for seq in prefix_seqs]
    prefix_seqs_indexed = [[word_to_index['<pad>']] * (maxlength - len(seq)) + seq for seq in prefix_seqs_indexed]

    return vocab, prefix_seqs_indexed, word_to_index, maxlength



class LSTMModel(nn.Module):
  def __init__(self, repr_dim, hidden_size, learning_rate, momentum, epsilon, vocab):
    super().__init__()
    self.repr_dim = repr_dim
    self.hidden_size = hidden_size
    self.model = None
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.lr = learning_rate
    self.momentum = momentum
    self.epsilon = epsilon

    self.embedding_layer = nn.Embedding(len(vocab), self.repr_dim, device=self.device)
    self.lstm = nn.LSTM(input_size=self.repr_dim, hidden_size=self.hidden_size, batch_first=True)
    self.output = nn.Linear(self.hidden_size, len(vocab), bias=False, device=self.device)
    self.to(self.device)

  def forward(self, contexts):
    embeddings = self.embedding_layer(contexts)
    hidden_reps, _ = self.lstm(embeddings)
    output = self.output(hidden_reps[:,-1])
    return output



# def train_lstm_model(trainset, repr_dim=100, hidden_size=256, learning_rate=0.1, momentum=0.9, epsilon=1e-5, num_epochs=100, batch_size=512):
def train_lstm_model(model,context_word_batches):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=model.lr, momentum=model.momentum)

    prev_loss = -math.inf
    for epoch in range(85):
        ct = 0
        avg_epoch_loss = 0
        for i, (contexts, words) in enumerate(context_word_batches):
            optimizer.zero_grad()
            predicted = model.forward(contexts)
            target = words
            loss = criterion(predicted, target)
            avg_epoch_loss += loss.item()
            ct += 1
            loss.backward()
            optimizer.step()

            if i % 500 == 0:
                print(loss.item(), end=' | ')

        avg_epoch_loss /= ct
        print(f"Epoch {epoch}, Loss: {avg_epoch_loss} PrevLossL {prev_loss}")

        if abs(avg_epoch_loss - prev_loss) <= model.epsilon:
            break
        prev_loss = avg_epoch_loss

    return model


# def train_model_with_hyperparams(repr_dim, hidden_size, learning_rate, momentum, epsilon, vocab, train_batches, valid_batches, max_epochs=30):
#     lm = LSTMModel(repr_dim, hidden_size, learning_rate, momentum, epsilon, vocab)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(lm.parameters(), lr=learning_rate, momentum=momentum)

#     prev_loss = -math.inf
#     for epoch in range(max_epochs):
#         epoch_loss = 0
#         ct = 0
#         for i, (contexts, words) in enumerate(train_batches):
#             optimizer.zero_grad()
#             predicted = lm.forward(contexts)
#             target = words
#             loss = criterion(predicted, target)
#             epoch_loss += loss.item()
#             ct += 1
#             loss.backward()
#             optimizer.step()

#         epoch_loss /= ct
#         valid_loss = 0
#         ct = 0
#         with torch.no_grad():
#             for i, (contexts, words) in enumerate(valid_batches):
#                 predicted = lm.forward(contexts)
#                 target = words
#                 loss = criterion(predicted, target)
#                 valid_loss += loss.item()
#                 ct += 1

#         valid_loss /= ct
#         print(f"Epoch {epoch}, Train Loss: {epoch_loss}, Valid Loss: {valid_loss}")

#         if abs(valid_loss - prev_loss) <= epsilon:
#             break
#         prev_loss = valid_loss

#     valid_perplexity = get_perp(lm, valid_batches, word_to_index)

#     return lm, valid_perplexity


# def optimize_hyperparams(vocab, prefix_seqs_indexed, word_to_index, maxlength, max_epochs=30):
#     repr_dim_values = [50, 100, 150]
#     hidden_size_values = [50, 100, 150]
#     learning_rate_values = [0.001, 0.01, 0.1]
#     momentum_values = [0.0, 0.9, 0.99]
#     epsilon_values = [0.001, 0.01, 0.1]

#     hyperparam_values = itertools.product(repr_dim_values, hidden_size_values, learning_rate_values, momentum_values, epsilon_values)

#     best_valid_perplexity = float('inf')
#     best_lm = None
#     for hyperparams in hyperparam_values:
#         repr_dim, hidden_size, learning_rate, momentum, epsilon = hyperparams
#         print(f"Trying hyperparams: {hyperparams}")
#         train_batches = torch.split(torch.tensor(prefix_seqs_indexed), 256)
#         valid_batches = train_batches[:10]
#         train_batches = train_batches[10:]

#         lm, valid_perplexity = train_model_with_hyperparams(repr_dim, hidden_size, learning_rate, momentum, epsilon, vocab, train_batches, valid_batches, max_epochs=max_epochs)
#         if valid_perplexity < best_valid_perplexity:
#             best_valid_perplexity = valid_perplexity
#             best_lm = lm

#     torch.save(best_lm.state_dict(), "BestCheckpoint.pth")
#     return best_lm


def get_perp(model, sentence, word_to_index):
    if (len(sentence) == 0):
        return float("NaN")
    contexts = sentence[:, :-1]
    words = sentence[:, -1]
    prob_dists = model.forward(contexts)
    ct = 0
    for i, word in enumerate(words):
#         print(prob_dists[i][word])
        shifted_vec = prob_dists[i] - prob_dists[i].min()
        vec_sum = torch.sum(shifted_vec)
        prob_vec = shifted_vec / vec_sum
        prob = prob_vec[word]
#         print(prob)
        ct += 1
        product = 1
        product *= prob.item()
    try:
        return math.pow(abs(1/product),1/ct)
    except:
        lite = 1
        


vocab, prefix_seqs_indexed, word_to_index, maxlength = get_vocab("./split/TrainingCorpus_PP.txt")

batches = torch.split(torch.tensor(prefix_seqs_indexed), 256)
context_word_batches = []
for batch in batches:
  context_word_batches.append((batch[:, :-1], batch[:, -1]))


# loaded_model = LSTMModel(100, 150, 0.001, 0.9, 0.001, vocab)
# train_lstm_model(loaded_model,context_word_batches)
# torch.save(loaded_model.state_dict(), "FirstSave2.pth")

# Loading
loaded_model = LSTMModel(100, 150, 0.001, 0.9, 0.001, vocab)
loaded_model.load_state_dict(torch.load(sys.argv[1]))

sentence = input("Enter a Sentence: ")
tokens = sentence.split(" ")
if(len(tokens) > 34):
    print("Please enter string of length less than 34")
    exit()
indices = [word_to_index.get(w, word_to_index['<unk>']) for w in tokens] 
sentence = [indices[:i] for i in range(1,len(indices)+1)]
sentence = [ ([word_to_index['<pad>']]*(maxlength-len(prefix)) + prefix) for prefix in sentence]
sentence = torch.tensor(sentence)
temp = get_perp(loaded_model, sentence, word_to_index)
print(temp)
