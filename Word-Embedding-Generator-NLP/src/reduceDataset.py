import os
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import random
import numpy as np
from collections import defaultdict



# Deleting the reducedDataset.txt file if exists
file_path = "reducedDataset.txt"
if os.path.isfile(file_path):
    os.remove(file_path)
    print("File deleted.")
else:
    print("File does not exist.")



# Extracting 100000 (1 lakh) sentences
ct = 0
data = []
with open("reviews_Movies_and_TV.json") as file:
    for line in file:
        dict = eval(line)
        # print(dict['reviewText'])
        sentences = sent_tokenize(dict['reviewText'])
        # print(sentences)
        for sen in sentences:
            data.append(sen.lower())
            ct += 1
            if(ct >= 100000):
                break
        if(ct >= 100000):
            break



# Removing multiple spaces and seperating some puntuations
data = [re.sub(r"[^a-zA-Z0-9]+", " ", s) for s in data]
# data = [s.replace(",", " , ").replace(".", " . ") for s in data]
data = [" ".join(s.split()) for s in data]



# Removing less frequent words
def ViewDatasetFreq(data):
    freq = {}
    for sen in data:
        for word in sen.split():
            if word not in freq.keys():
                freq[word] = 1
            else:
                freq[word] += 1

    vocabulary = set()
    for sentence in data:
        for word in sentence.split():
            vocabulary.add(word)

    freq_ct = {}
    for word in vocabulary:
        if freq[word] not in freq_ct.keys():
            freq_ct[freq[word]] = 1
        else:
            freq_ct[freq[word]] += 1

    myKeys = list(freq_ct.keys())
    myKeys.sort()
    sorted_freq_ct = {i: freq_ct[i] for i in myKeys}
    print(sorted_freq_ct)
    print("--------------------")

ViewDatasetFreq(data)

freq = {}
for sen in data:
    for word in sen.split():
        if word not in freq.keys():
            freq[word] = 1
        else:
            freq[word] += 1

for i in range(len(data)):
    new_sen = ""
    for word in data[i].split():
        if(freq[word] > 3):
            new_sen += word
            new_sen += ' '
    data[i] = new_sen

ViewDatasetFreq(data)




# Sub-sampling for frequent stopwords
def subsample_stopwords(sentences, threshold=1e-5):
    # Get stop words and tokenize sentences
    stop_words = set(stopwords.words('english'))
    tokenized_sentences = [sentence.split() for sentence in sentences]

    # Count word frequencies and remove non-stop words
    freq = defaultdict(int)
    for sentence in tokenized_sentences:
        for word in sentence:
            if word in stop_words:
                freq[word] += 1
    total_words = sum(freq.values())

    # Compute the probability of keeping each stop word
    stopword_probs = {word: 1 - np.sqrt(threshold / (freq[word] / total_words)) for word in freq}

    # Subsample stop words in each sentence
    subsampled_sentences = []
    for sentence in tokenized_sentences:
        subsampled_sentence = []
        for word in sentence:
            if word in stop_words and random.random() < stopword_probs[word]:
                subsampled_sentence.append(word)
            elif word not in stop_words:
                subsampled_sentence.append(word)
        if subsampled_sentence:
            subsampled_sentences.append(" ".join(subsampled_sentence))

    return subsampled_sentences

data = subsample_stopwords(data)

ViewDatasetFreq(data)



# saving the reducedDataset
with open("reducedDataset.txt", "w") as file:
    for line in data:
        file.write(line+" \n")
