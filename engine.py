'''
This is a character level auto-regressive model

1. Getting the data 
2. Constructing a vocabulary
3. Tokenizing the data 
4. Constructing the model 
5. defining the loss and optimizer 
6. Training loop 
7. Evaluating the model
'''
import torch 
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt
import random

import wget
import os
from typing import List

# Context length will determine, how many characters will be used to predict the next character
CONTEXT_LENGTH  = 3
MINOR_DEBUG = True
MAJOR_DEBUG = False
random.seed(42)
generator = torch.Generator().manual_seed(2147483647)

# Useful Functions 
def get_data(url: str, file_name: str) -> List[str]:
    try:
        if not os.path.exists(file_name):
            print(f"Downloading {file_name}...")
            wget.download(url, file_name)
            print("\nDownload complete!")
        else:
            print(f"File {file_name} already exists, using local copy")
        
        with open(file_name, 'r') as file:
            corpus = file.read().splitlines()
            
        return corpus
        
    except Exception as e:
        raise Exception(f"Error processing file: {e}")

def construct_vocab(corpus):
    one_big_blob = ''.join(corpus)
    unique_chars = set(one_big_blob)
    list_unique_chars = list(unique_chars)
    vocab = sorted(list_unique_chars)
    return vocab

def construct_mapping(vocab):
    element_to_index = {element:index+1 for index,element in enumerate(vocab)}
    # add the special token '.'
    element_to_index['.'] = 0 
    # index_to_element 
    index_to_element = {value:key for key,value in element_to_index.items()}
    return element_to_index, index_to_element
    
def construct_dataset(corpus: List[str], index_to_element, element_to_index):
    X , Y = [], []
    # get each word (Corpus is a list of words: view the get_data function for more info) 
    for word in corpus:
        context = [0] * CONTEXT_LENGTH
        for char in word+'.':
            X.append(context) # 0 0 0 
            Y.append(element_to_index[char])
            if MAJOR_DEBUG:
                print(''.join(index_to_element[i] for i in context), '-->', index_to_element[Y[-1]])
            context = context[1:] + [element_to_index[char]] # sliding window 
            '''
            word: 'hello'
            context: [0,0,0]
            X: [0,0,0]
            Y: [h]

            context: [0,0,h]
            X: [0,0,h]
            Y: [e]

            context: [0,h,e]
            X: [0,h,e]
            Y: [l]

            context: [h,e,l]
            X: [h,e,l]
            Y: [l]

            context: [e,l,l]
            X: [e,l,l]  
            Y: [l]

            context: [l,l,o]
            X: [l,l,o]
            Y: [.]
            '''

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    if MINOR_DEBUG:
        print(X.shape, Y.shape)
    return X, Y

def split(corpus, index_to_element, element_to_index):
    '''
    Return the training, dev and test sets
    '''
    n1 = int(0.8*len(corpus))
    n2 = int(0.9*len(corpus))
    Xtr, Ytr = construct_dataset(corpus[:n1], index_to_element, element_to_index)
    Xdev, Ydev = construct_dataset(corpus[n1:n2], index_to_element, element_to_index)
    Xte, Yte = construct_dataset(corpus[n2:], index_to_element, element_to_index)
    return Xtr, Ytr, Xdev, Ydev, Xte, Yte

def gen_model_params():
    '''
    Model Architecture: A Neural Probabilistic Language Model 
    Authors: Yoshua Bengio et al
    '''
    pass 

########## Get the data #########
corpus = get_data("https://raw.githubusercontent.com/karpathy/makemore/master/names.txt", "names.txt")

######### Construct a vocabulary for the model using the corpus #########
vocab = construct_vocab(corpus)

element_to_index, index_to_element = construct_mapping(vocab)

########## Build the dataset ##########
'''
So given the corpus of data, we will build a dataset: X -> Y mapping 
Our model will learn to predict the next character given a context 'CONTEXT_LENGTH' characters long 
We will divide the data into training, dev, test sets.
'''
Xtr, Ytr, Xdev, Ydev, Xte, Yte = split(corpus, index_to_element, element_to_index)