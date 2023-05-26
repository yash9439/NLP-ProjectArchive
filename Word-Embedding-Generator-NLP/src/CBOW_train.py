import numpy as np
import torch
import random
from torch.utils.data import Dataset
np.random.seed(23)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm



class Word2vecDataLoader(Dataset):
    def __init__(self, window_size, inputFileName, min_count):
        self.window_size = window_size
        self.input_file = open(inputFileName, encoding="utf8")

        self.negative_words = []
        self.discards = []

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.inputFileName = inputFileName
        self.read_words(min_count)

    def __len__(self):
        return self.sentences_count

    def __getitem__(self, idx):
        while True:
            line = self.input_file.readline()
            # Check if Line
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()
            if len(line) > 1:
                words = line.split()
                if len(words) > 1:
                    word_ids = [self.word2id[w] for w in words if w in self.word2id]

                    boundary = self.window_size//2 #np.random.randint(1, self.window_size)
                    cbow_data = []
                    for i, v in enumerate(word_ids):
                        tmp_lst = []
                        for u in word_ids[max(i - boundary, 0):i + boundary + 1]:# as python handles max indexing case so no need to take min
                            if u!=v:
                                tmp_lst.append(u)
                        if len(tmp_lst) < 2*boundary:
                            tmp_lst += [0]*(2*boundary - len(tmp_lst))
                        cbow_data.append((tmp_lst, v, self.get_negative_samples(tmp_lst, 10)))
                    return cbow_data
    
    
    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="utf8"):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print("Total Words Read So far: " + str(int(self.token_count / 1000000)) + "M words.")
        w_id = 1
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = w_id
            self.id2word[w_id] = w
            self.word_frequency[w_id] = c
            w_id += 1
        self.word2id["<SPACE>"] = 0
        self.id2word[0] = "<SPACE>"
        self.word_frequency[0] = 1
        print("Total embeddings: " + str(len(self.word2id)))
    
    def get_negative_samples(self, context, k):
        negatives = []
        while len(negatives) < k:
            word = random.randint(0, len(self.word2id)-1)
            if word not in context:
                negatives.append(word)
        return negatives

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]
        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)




class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size, num_neg_samples):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output_weights = nn.Embedding(vocab_size, embed_dim)
        self.context_size = context_size
        self.num_neg_samples = num_neg_samples

    def forward(self, context, target, neg_samples):
        context_embeds = self.embeddings(context)  # (batch_size, context_size, embed_dim)
        avg_embeds = torch.mean(context_embeds, dim=1)  # (batch_size, embed_dim)
        target_embeds = self.output_weights(target)  # (batch_size, embed_dim)

        pos_scores = torch.sum(avg_embeds * target_embeds, dim=1)  # (batch_size,)
        pos_loss = -torch.log(torch.sigmoid(pos_scores)).mean()

        neg_embeds = self.output_weights(neg_samples)  # (batch_size, num_neg_samples, embed_dim)
        neg_scores = torch.bmm(neg_embeds, avg_embeds.unsqueeze(2)).squeeze()  # (batch_size, num_neg_samples)
        neg_loss = -torch.log(torch.sigmoid(-neg_scores)).mean()

        return pos_loss + neg_loss
    
    def save_embedding(self, id2word, file_name):
        embedding = self.embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
#             f.write('%d %d\n' % (len(id2word), self.embeddings.emb_dimension))
            for w_id, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[w_id]))
                f.write('%s %s\n' % (w, e))

def train_cbow(data, vocab_size, embed_dim, context_size, num_neg_samples, num_epochs, lr):
    model = CBOW(vocab_size, embed_dim, context_size, num_neg_samples)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        for context, target, neg_samples in tqdm(data):
            optimizer.zero_grad()
            loss = model(context, target, neg_samples)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data)}")
    model.save_embedding(dataset.id2word, output_file_name)
    torch.save(model.state_dict(), 'my_model.pth')

input_file = "/kaggle/input/reduceddataset/reducedDataset.txt"
output_file = "word_embeddings.txt"
emb_dimension=350
batch_size=128
window_size=4
iterations=150
initial_lr=0.001
min_count=3

dataset = Word2vecDataLoader(window_size, input_file, min_count)
dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=False, num_workers=20, collate_fn=dataset.collate)
output_file_name = output_file
emb_size = len(dataset.word2id)

data = []
for i, sample_batched in enumerate(tqdm(dataloader)):
    if len(sample_batched[0]) > 1:
        pos_u = sample_batched[0]
        pos_v = sample_batched[1]
        neg_v = sample_batched[2]
        new_tuple = (pos_u,pos_v,neg_v)
        data.append(new_tuple)

train_cbow(data, len(dataset.word2id), embed_dim=100, context_size=4, num_neg_samples=5, num_epochs=10, lr=0.001)

