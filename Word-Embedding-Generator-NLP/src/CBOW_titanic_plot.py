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
words = ['titanic']

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

# Convert the dictionary into data array
lst = []
for word in top_words['titanic']:
    lst.append(word_vectors_dict[word])

print(len(lst))
data = np.array(lst)
print(len(data))


# Compute the t-SNE embedding
tsne = TSNE(n_components=2, perplexity=5, learning_rate=200)
tsne_embedding = tsne.fit_transform(data)


plt.figure(figsize=(8, 8))
plt.scatter(tsne_embedding[:10, 0], tsne_embedding[:10, 1], label='titanic', color='red')
for i, word in enumerate(top_words['titanic']):
    plt.text(tsne_embedding[i, 0], tsne_embedding[i, 1], word)
 
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.show()
