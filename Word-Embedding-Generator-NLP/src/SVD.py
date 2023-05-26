import os
import numpy as np
from sklearn.decomposition import TruncatedSVD

corpus = []

# Deleting the word_vectors.txt file if exists
file_path = "word_vectors.txt"
if os.path.isfile(file_path):
    os.remove(file_path)
    print("File deleted.")
else:
    print("File does not exist.")


with open("reducedDataset.txt") as file:
    for line in file:
        corpus.append(line.strip())



# Define the window size and vector dimensionality
window_size = 2
vector_size = 100



# Create a vocabulary and a co-occurrence matrix
print("Generating Vocabulary")
vocabulary = set()
for sentence in corpus:
    for word in sentence.split():
        vocabulary.add(word)
print("Vocabulary Generated")


vocabulary = list(vocabulary)
word_to_id = {word: i for i, word in enumerate(vocabulary)}
cooccurrence_matrix = np.zeros((len(vocabulary), len(vocabulary)))


print("Generating Cooccurrence_Matrix")
for sentence in corpus:
    words = sentence.split()
    for i, word in enumerate(words):
        for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
            if i != j:
                cooccurrence_matrix[word_to_id[word], word_to_id[words[j]]] += 1
print("Cooccurrence_Matrix Generated")




# Apply SVD to the co-occurrence matrix
print("Applying SVD")
svd = TruncatedSVD(n_components=vector_size)
word_vectors_svd = svd.fit_transform(cooccurrence_matrix)
print("SVD Applied")





# Normalize the word vectors
print("Normalizing Vector")
word_vectors_normalized = word_vectors_svd / np.linalg.norm(word_vectors_svd, axis=1, keepdims=True)
print("Normalization Completed")




# Create a dictionary of word vectors
word_vectors_dict = {}
for i, word in enumerate(vocabulary):
    word_vectors_dict[word] = word_vectors_normalized[i]




# Save the word vectors to a file
print("Saving Word Embeddings")
with open('word_vectors.txt', 'w') as f:
    f.write('{} {}\n'.format(len(vocabulary), vector_size))
    for word, vector in word_vectors_dict.items():
        vector_str = ' '.join([str(x) for x in vector])
        f.write('{} {}\n'.format(word, vector_str))


with open("vocab.txt", "w") as f:
    for item in vocabulary:
        f.write(str(item) + "\n")


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# # Define the words for which we want to plot the word vectors
words = ['guitarists', 'prefer', 'childish', 'selling', 'selfishness']

# Get the top-10 most similar words for each word
top_words = {}
for word in words:
    word_vector = word_vectors_dict[word]
    cosine_similarities = np.dot(word_vectors_normalized, word_vector)
    most_similar_indices = np.argsort(cosine_similarities)[-11:-1]  # exclude the word itself
    most_similar_words = [vocabulary[i] for i in most_similar_indices]
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
