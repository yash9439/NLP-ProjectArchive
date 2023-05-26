import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_embeddings(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    word2idx = {}
    embeddings = {}
    vocab = set()
    
    for line in lines[1:]:
        fields = line.strip().split()
        word = fields[0]
        vector = list(map(float, fields[1:]))
        word2idx[word] = len(word2idx)
        embeddings[word] = vector
        vocab.add(word)
    vocab = list(vocab)
    return vocab, word2idx, embeddings


vocab, word2idx, word_vectors_dict = load_embeddings('word_embeddings.txt')

word_vectors_normalized = []
for i, word in enumerate(vocab):
    word_vectors_normalized.append(word_vectors_dict[word])


# # Define the words for which we want to plot the word vectors
words = ['guitarists', 'prefer', 'childish', 'selling', 'selfishness']

print(len(vocab))
# print(vocab[1579])

# Get the top-10 most similar words for each word
top_words = {}
for word in words:
    word_vector = word_vectors_dict[word]
    cosine_similarities = np.dot(word_vectors_normalized, word_vector)
    most_similar_indices = np.argsort(cosine_similarities)[-11:-1]  # exclude the word itself
    most_similar_words = [vocab[i] for i in most_similar_indices]
    top_words[word] = most_similar_words[::-1]  # reverse the list

print(top_words)

# Convert the dictionary into 5 data arrays
lst = []
for word in top_words['guitarists']:
    lst.append(word_vectors_dict[word])
data1 = np.array(lst)

lst = []
for word in top_words['prefer']:
    lst.append(word_vectors_dict[word])
data2 = np.array(lst)

lst = []
for word in top_words['childish']:
    lst.append(word_vectors_dict[word])
data3 = np.array(lst)

lst = []
for word in top_words['selling']:
    lst.append(word_vectors_dict[word])
data4 = np.array(lst)

lst = []
for word in top_words['selfishness']:
    lst.append(word_vectors_dict[word])
data5 = np.array(lst)

lst = []
for word in words:
    lst.append(word_vectors_dict[word])
data6 = np.array(lst)

# Concatenate the data arrays into one array
data = np.concatenate((data1, data2, data3, data4, data5, data6))


# Compute the t-SNE embedding
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
tsne_embedding = tsne.fit_transform(data)




plt.figure(figsize=(8, 8))
plt.scatter(tsne_embedding[:10, 0], tsne_embedding[:10, 1], label='guitarists', color='red')
for i, word in enumerate(top_words['guitarists']):
    plt.text(tsne_embedding[i, 0], tsne_embedding[i, 1], word)

plt.scatter(tsne_embedding[10:20, 0], tsne_embedding[10:20, 1], label='prefer', color='yellow')
for i, word in enumerate(top_words['prefer']):
    plt.text(tsne_embedding[i+10, 0], tsne_embedding[i+10, 1], word)

plt.scatter(tsne_embedding[20:30, 0], tsne_embedding[20:30, 1], label='childish', color='grey')
for i, word in enumerate(top_words['childish']):
    plt.text(tsne_embedding[i+20, 0], tsne_embedding[i+20, 1], word)

plt.scatter(tsne_embedding[30:40, 0], tsne_embedding[30:40, 1], label='selling', color='blue')
for i, word in enumerate(top_words['selling']):
    plt.text(tsne_embedding[i+30, 0], tsne_embedding[i+30, 1], word)

plt.scatter(tsne_embedding[40:50, 0], tsne_embedding[40:50, 1], label='selfishness', color='black')
for i, word in enumerate(top_words['selfishness']):
    plt.text(tsne_embedding[i+40, 0], tsne_embedding[i+40, 1], word)

plt.scatter(tsne_embedding[50:55, 0], tsne_embedding[50:55, 1], label='Chosen Words', color='orange')
for i, word in enumerate(words):
    plt.text(tsne_embedding[i+50, 0], tsne_embedding[i+50, 1], word)
 
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.show()
