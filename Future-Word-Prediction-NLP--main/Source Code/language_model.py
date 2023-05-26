import nltk
nltk.download('punkt') 
from nltk.tokenize import sent_tokenize
import sys
import math
import random
from cleaning import tokenize
import argparse
import re
import numpy

def write_array_to_file(file_name, array):
    with open(file_name, 'w') as file:
        for item in array:
            file.write(item + '\n')

def read_file_to_string(file_name):
    with open(file_name, 'r') as file:
        return file.read()


def GenerateDataSet(Corpus_path, TrainingCorpus_path, TestingCorpus_path, devCorpus_path):
    # INPUT: 3 locations in the directory
    # OUTPUT: 2 files generated
    # dividing the corpus into training and testing dataSet

    # Storing the coupus
    Corpus = read_file_to_string(Corpus_path)
    Corpus = sent_tokenize(Corpus)
    # print(Corpus)
    ct = 0
    for i in Corpus:
        Corpus[ct] = tokenize(i)
        ct += 1
    print(Corpus)

    # Dividing the Corpus into training and testing Data
    random.shuffle(Corpus)
    TestingDataLines = Corpus[:1000]
    DevDataLines = Corpus[1000:2000]
    TrainingDataLines = Corpus[2000:]

    write_array_to_file(TrainingCorpus_path, TrainingDataLines)
    write_array_to_file(devCorpus_path, DevDataLines)
    write_array_to_file(TestingCorpus_path, TestingDataLines)


def GenerateNgrams(sentences_plaintext, n):
    ''' INPUT: "he is a good
               boy who likes candy" , 3
     OUTPUT: "he is a": 1
              "is a good" : 1
              "boy who likes" : 1
              "who likes candy" : 1 
    '''
    # Corpus is plain text
    # n is the n gram

    ngrams_dic = dict()
    for sentence in sentences_plaintext:
        words = re.findall(r'\b\S+\b', sentence)
        if(len(words) > 1):
            temp  = []
            for i in range(n-1):
                temp.append("<SOS>")
            for word in words:
                temp.append(word)
            ngram_list = []
            for i in range(n-1,len(temp)):
                ngram_single_text_list = []
                for j in range(n):
                    ngram_single_text_list.append(temp[i-n+1+j])
                ngram_single_text_tuple = tuple(ngram_single_text_list)
                ngram_list.append(ngram_single_text_tuple)
            for tupl in ngram_list:
                if tupl not in ngrams_dic:
                    ngrams_dic[tupl] = 1
                else:
                    ngrams_dic[tupl] += 1
    return ngrams_dic


def GenerateAllNgrams(sentences_plaintext, n, limit):
    ''' INPUT: "he is a good
               boy who likes candy" , 3
     OUTPUT: [n] = list of ngrams
    '''
    ngram_allDic = dict()
    for i in range(2,n+1):
        ngram_allDic[i] = GenerateNgrams(sentences_plaintext,i)
    unigram = GenerateNgrams(sentences_plaintext,1)
    unigram_res = unigram.copy()
    unigram_res[('<UNK>',)] = 0
    for key,value in unigram.items():
        try:
            if(value <= limit):
                unigram_res[('<UNK>',)] += value
                unigram_res.pop(key)
            else:
                if key not in unigram_res:
                    unigram_res[key] = 1
                else:
                    unigram_res[key] += value
        except:
            lite = 1
    ngram_allDic[1] = unigram_res
    return ngram_allDic


# count no. of words in the given string.
def Count_HighestOrder(string, ngram_allDic):
    if len(string) == 0:
        return 0
    n = len(re.findall(r'\b\S+\b', string))
    # Retieve the count of that string from a central dict.
    try:
        return ngram_allDic[n][tuple(re.findall(r'\b\S+\b', string))]
    except:
        return 0


# sum of counts of Summation(history + vi).
def sameHistoryCount_HighestOrder(history, ngram_allDic):
    cnt = 0
    n = len(re.findall(r'\b\S+\b', history)) + 1
    n = min(n,4)
    for key, value in ngram_allDic[n].items():
        chk = key[:-1]
        # for i in range(n-1):
        #     chk.append(key[i])
        if tuple(chk) == tuple(re.findall(r'\b\S+\b', history)):
            cnt += value
    return cnt


def Count_LowerOrder(history, string, n, ngram_allDic):
    term = history
    term += " "
    term += string
    try:
        if tuple(re.findall(r'\b\S+\b', term)) in ngram_allDic[n].keys():
            return 1
    except:
        return 0
    return 0


def UniqueCount_history(history, ngram_allDic):
    # No. of gram of Summation (history + vi) exisiting.
    cnt = 0
    n = len(re.findall(r'\b\S+\b', history)) + 1
    # print(n)
    for key, value in ngram_allDic[n].items():
        chk = []
        for i in range(n-1):
            chk.append(key[i]) 
        # print(chk)
        if tuple(chk) == tuple(re.findall(r'\b\S+\b', history)):
            cnt += 1
        # else:
        #     with open("bug.txt", "a") as bug:
        #         bug.write(chk[0] +" " + chk[1] + "\n")
        #     bug.close()
    return cnt

def total(ngram_allDic,idx):
    cnt = 0
    for key, value in ngram_allDic[idx].items():
        cnt += value
    return cnt


def Variable_history(current,n , ngram_allDic):
    # No. of Variable history
    cnt = 0
    for key, value in ngram_allDic[n].items():
        if(key[n-1] == current):
            cnt += value
    return cnt


#  INPUT =  history and current are string
#  OUTPUT = P(current/history)
def kneserNey(history, current, highestOrder, ngram_allDic):
    n = len(re.findall(r'\b\S+\b', history))+1

    # # Checking if current word is in vocabulary
    # if tuple([current]) not in ngram_allDic[1].keys():
    #     return 0.75/ngram_allDic[1][('<UNK>',)]
    
    # Recusion Base Case
    if n == 1:
        if tuple([current]) in ngram_allDic[1].keys():
            return (1-0.75)/len(ngram_allDic[1]) + 0.75*ngram_allDic[1][('<UNK>',)]/len(ngram_allDic[1])
        else:
            return 0 + 0.75*ngram_allDic[1][('<UNK>',)]/len(ngram_allDic[1])

    if highestOrder == True:
        try:
            term = history
            term += " "
            term += current
            FirstTerm = max(Count_HighestOrder(term, ngram_allDic) - 0.75, 0)/sameHistoryCount_HighestOrder(history, ngram_allDic)
        except:
            FirstTerm = 0
    else:
        try:
            FirstTerm = max(Count_LowerOrder(history ,current, n, ngram_allDic) - 0.75, 0)/len(ngram_allDic[n])
        except:
            FirstTerm = 0

    try:
        Lambda = (0.75/sameHistoryCount_HighestOrder(history, ngram_allDic))*UniqueCount_history(history, ngram_allDic)
    except:
        # print(0.75/ngram_allDic[1][('<UNK>',)])
        return 0.75*ngram_allDic[1][('<UNK>',)]/len(ngram_allDic[1])
    
    # Doing Reccursion Call
    new_history = ""
    tmp = re.findall(r'\b\S+\b', history)
    for i in range(1,len(tmp)):
        new_history += tmp[i]
        new_history += " "
    SecondTerm = Lambda*kneserNey(new_history, current, False, ngram_allDic)

    return FirstTerm + SecondTerm



#  INPUT =  history and current are string
#  OUTPUT = P(current/history)
def wittenBell(history, current, ngrams_allDict):
    n = len(re.findall(r'\b\S+\b', history)) + 1
    # Recusion Base Case
    if n == 1:
        try:
            return Count_HighestOrder(current, ngrams_allDict)+1/(total(ngram_Alldict,1)+len(ngram_Alldict[1]))
        except:
            return 1/len(ngrams_allDict[1])

    # Define Lambda
    try:
        Lambda = Variable_history(current,n, ngrams_allDict)/(Variable_history(current,n, ngrams_allDict) + sameHistoryCount_HighestOrder(history, ngrams_allDict))
    except:
        # print(history)
        Lambda = 1/len(ngrams_allDict[n])

    # The first term in WITTEN-Bell expression and it's exception is handled above.
    str = history
    str += " "
    str += current
    try:
        FirstTerm = Count_HighestOrder(str, ngrams_allDict)/(sameHistoryCount_HighestOrder(history, ngrams_allDict)+Count_HighestOrder(str, ngrams_allDict))
    except:
        FirstTerm = 1/len(ngram_Alldict[n])
    
    new_history = ""
    tmp = re.findall(r'\b\S+\b', history)
    for i in range(1,len(tmp)):
        new_history += tmp[i]
        new_history += " "
    return FirstTerm*(Lambda) + (1-Lambda)*wittenBell(new_history, current, ngrams_allDict)


def perplexity(listOfSentence,model,ngram_Alldict):
    if model == 'w':
        for sentence in listOfSentence:
            product = 1
            count = 0
            words = re.findall(r'\b\S+\b', sentence)
            for i in range(len(words)):
                history = ""
                current = words[i]
                for j in range(i-3,i):
                    if(j < 0):
                        history += "<SOS> "
                    else:
                        history += words[j]
                        history += " "
            return wittenBell(history,current,ngram_Alldict)
    elif model == 'k':
        for sentence in listOfSentence:
            product = 1
            count = 0
            words = re.findall(r'\b\S+\b', sentence)
            for i in range(len(words)):
                history = ""
                current = words[i]
                for j in range(i-3,i):
                    if(j < 0):
                        continue
                    history += words[j]
                    history += " "
            return kneserNey(history,current,True,ngram_Alldict)



def perplexity2(listOfSentence,model,ngram_Alldict,):
    perplexity = []
    if model == 'w':
        for sentence in listOfSentence:
            product = 1
            count = 0
            words = re.findall(r'\b\S+\b', sentence)
            for i in range(len(words)):
                history = ""
                current = words[i]
                for j in range(i-3,i):
                    if(j < 0):
                        history += "<SOS> "
                    else:
                        history += words[j]
                        history += " "
                product *= wittenBell(history,current,ngram_Alldict,)
                count += 1
            try:
                tempo = pow(1/product,1/count)
                perplexity.append(tempo)
                print(tempo)
                with open("2021114012_LM2_test-perplexity.txt", "a") as bug:
                    bug.write(sentence + "\t" + str(tempo) + "\n")
                bug.close()
            except:
                lite = 1
    elif model == 'k':
        for sentence in listOfSentence:
            product = 1
            count = 0
            words = re.findall(r'\b\S+\b', sentence)
            for i in range(len(words)):
                history = ""
                current = words[i]
                for j in range(i-3,i):
                    if(j < 0):
                        continue
                    history += words[j]
                    history += " "
                product *= kneserNey(history,current,True,ngram_Alldict)
                count += 1
            try:
                tempo = pow(1/product,1/count)
                perplexity.append(tempo)
                print(tempo)
                with open("2021114012_LM1_test-perplexity.txt", "a") as bug:
                    bug.write(sentence + "\t" + str(tempo) + "\n")
                bug.close()
            except:
                lite = 1
    else:
        return -404
    avg = 0
    for i in perplexity:
        avg += i
    with open("avg.txt", "a") as av:
        av.write(avg/len(perplexity))
    av.close()
    return avg/len(perplexity)
    # return perplexity


if __name__ == '__main__':
    # GenerateDataSet("../Pride-and-Prejudice-Jane-Austen.txt","./TrainingCorpus_PP.txt","./TestingCorpus_PP.txt","./DevCorpus_PP.txt")
    # GenerateDataSet("../Ulysses-James-Joyce.txt","./TrainingCorpus_UJJ.txt","./TestingCorpus_UJJ.txt","./DevCorpus_UJJ.txt")

    if(sys.argv[1] == 'k'):
        corpus = read_file_to_string(sys.argv[2])
        corpus = corpus.split('\n')
        ngram_Alldict = (GenerateAllNgrams(corpus,4,1))
        sen = input("Enter a Sentence: ")
        print(perplexity([sen],'k',ngram_Alldict))
    elif(sys.argv[1] == 'w'):
        corpus = read_file_to_string(sys.argv[2])
        corpus = corpus.split('\n')
        ngram_Alldict = (GenerateAllNgrams(corpus,4,1))
        sen = input("Enter a Sentence: ")
        print(perplexity([sen],'w',ngram_Alldict))
    else:
        print("Error in argument")
    
    