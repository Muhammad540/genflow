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
import torch.nn.functional as F
import random

import wget
import os
from typing import List

# Context length will determine, how many characters will be used to predict the next character
CONTEXT_LENGTH  = 5
BATCH_SIZE = 32
EPOCHS = 200000
LEARNING_RATE = 0.1
HIDDEN_DIM = 200
EMBEDDING_DIM = 10
VOCAB_SIZE = 0  # Will be set after vocabulary construction
MINOR_DEBUG = True
MAJOR_DEBUG = False
torch.manual_seed(2147483647)
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
    vocab_size = len(vocab)  # Use local variable instead
    if MINOR_DEBUG:
        print("Vocabulary: ", vocab)
    return vocab, vocab_size

def construct_mapping(vocab):
    element_to_index = {element:index for index,element in enumerate(vocab)}
    # Remove space if it exists in the original vocab
    if ' ' in element_to_index:
        del element_to_index[' ']
    # add space as the special token at the end of the vocab
    element_to_index[' '] = len(element_to_index) 
    # index_to_element 
    index_to_element = {value:key for key,value in element_to_index.items()}
    if MINOR_DEBUG:
        print("Vocabulary size:", len(vocab))
        print("Number of indices:", len(element_to_index))
        print("First few mappings:", dict(list(element_to_index.items())[:5]))
        print("First few reverse mappings:", dict(list(index_to_element.items())[:5]))
        
    return element_to_index, index_to_element
    
def construct_dataset(corpus: List[str], index_to_element, element_to_index):
    X , Y = [], []
    # get each word (Corpus is a list of words: view the get_data function for more info) 
    for word in corpus:
        context = [0] * CONTEXT_LENGTH
        for char in word+' ':
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

def gen_model_params(embedding_dim: int, hidden_dim: int, context_length: int, vocab_size: int, Kaiming_init: bool = True, batch_norm: bool = True):
    '''
    Model Architecture: A Neural Probabilistic Language Model 
    Authors: Yoshua Bengio et al
    '''
    C = torch.randn((vocab_size+1, embedding_dim))
    if Kaiming_init:
        W1  = torch.randn((embedding_dim*context_length, hidden_dim)) * (5/3)/((embedding_dim*context_length)**0.5)
    else:
        W1 = torch.randn((embedding_dim*context_length, hidden_dim))
    W2 = torch.randn((hidden_dim, vocab_size+1)) * 0.01 # This is to prevent model from being overly confident in predicting a wrong character at the start of the training
    b2 = torch.randn((vocab_size+1)) * 0 # At the start of training, we dont want the model to favour any particular character over others rather every character is equally likely 1/size_vocab    
    
    # Batch Normalization Params 
    bngain = torch.ones((1, hidden_dim)) # kinda like variance to provide some flexibility for the model to learn 
    bnbias = torch.zeros((1, hidden_dim)) 
    bnmean_running = torch.zeros((1,hidden_dim))
    bnstd_running = torch.ones((1,hidden_dim))
    
    if not batch_norm:
        b1 = torch.randn((hidden_dim))
        parameters = [C, W1, b1, W2, b2]
        return parameters 
    else:
        parameters = [C, W1, W2, b2, bngain, bnbias]
        return parameters, bnmean_running, bnstd_running

def enable_gradients(parameters):
    for p in parameters:
        p.requires_grad = True
    return parameters

def training_loop(parameters, bnmean_running, bnstd_running, batch_size, epochs, learning_rate, Xtr, Ytr, batch_norm: bool = True):
    if batch_norm:
        C, W1, W2, b2, bngain, bnbias = parameters
    else:
        C, W1, b1, W2, b2 = parameters
    steps = []
    losses = []
    for i in range(epochs):
        # construct the batch 
        index = torch.randint(0, Xtr.shape[0], (batch_size,))
        
        # text embedding 
        embedding = C[Xtr[index]] # Embedding matrix is a lookup table for the characters; which we learn from the data 
        embedding_concat = embedding.view(batch_size, -1)
        
        # forward pass 
        if batch_norm:
            hpreact = embedding_concat @ W1
            bnmeani = hpreact.mean(0, keepdim=True)
            bnstd = hpreact.std(0, keepdim=True)
            hpreact = bngain * (hpreact - bnmeani) / bnstd + bnbias
            with torch.no_grad():
                bnmean_running = bnmean_running * 0.999 + bnmeani * 0.001
                bnstd_running = bnstd_running * 0.999 + bnstd * 0.001
        else:
            hpreact = embedding_concat @ W1 + b1
        
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Ytr[index])
        
        # backprop 
        for p in parameters:
            p.grad = None
        loss.backward()
        
        # update
        if i > 100000:
            learning_rate = 0.01
        for p in parameters:
            p.data -= learning_rate * p.grad
        
        steps.append(i)
        losses.append(loss.item())
        
        if i % 10 == 0:
            print(f"Epoch {i}, Loss: {loss.item()}")
        
    return steps, losses, bnmean_running, bnstd_running
    
def evaluate_model(parameters, split, Xtr, Ytr, Xdev, Ydev, Xte, Yte, batch_norm: bool = True, bnmean_running=None, bnstd_running=None):
    if batch_norm:
        C, W1, W2, b2, bngain, bnbias = parameters
    else:
        C, W1, b1, W2, b2 = parameters
    x,y = {
        'train': (Xtr, Ytr),
        'dev': (Xdev, Ydev),
        'test': (Xte, Yte)
    }[split]
    if batch_norm:
        # (N, context_length, embedding_dim)
        embedding = C[x]
        # (N, embedding_dim * context_length)
        hpreact = embedding.view(x.shape[0], -1) @ W1
        hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    else:
        embedding = C[x]
        hpreact = embedding.view(x.shape[0], -1) @ W1 + b1
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    return split, loss.item()

def plot_loss(steps, losses):
    plt.plot(steps, losses)
    plt.savefig('loss_plot.png')

def sample_from_model(parameters, context_length, num_generation, index_to_element, batch_norm: bool = True, bnmean_running=None, bnstd_running=None):
    if batch_norm:
        C, W1, W2, b2, bngain, bnbias = parameters
    else:
        C, W1, b1, W2, b2 = parameters

    for _ in range(num_generation):
        generated_text = []
        context = [0] * context_length
        while True:
            if batch_norm:
                embedding = C[torch.tensor([context])]
                hpreact = embedding.view(1, -1) @ W1
                hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
            else:
                embedding = C[torch.tensor([context])]
                hpreact = embedding.view(1, -1) @ W1 + b1
            h = torch.tanh(hpreact)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            next_char_index = torch.multinomial(probs, num_samples=1).item()
            
            if next_char_index not in index_to_element:
                continue
            
            context = context[1:] + [next_char_index]
            generated_text.append(next_char_index)
            if next_char_index == len(index_to_element) - 1:
                break
            
        valid_chars = [index_to_element[i] for i in generated_text if i in index_to_element]
        print(''.join(valid_chars))

def compute_batch_norm(parameters, Xtr):
    C, W1, W2, b2, bngain, bnbias = parameters
    embedding = C[Xtr]
    hpreact = embedding.view(Xtr.shape[0], -1) @ W1
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstd = hpreact.std(0, keepdim=True)
    return bnmeani, bnstd

# I delibrately used alot of functional programming to make the code more readable and modular 
########## Get the data #########
corpus = get_data("https://raw.githubusercontent.com/danielmiessler/SecLists/refs/heads/master/Passwords/Common-Credentials/10-million-password-list-top-10000.txt", "passwords.txt")

######### Construct a vocabulary for the model using the corpus #########
vocab, VOCAB_SIZE = construct_vocab(corpus)

if MINOR_DEBUG:
    print(VOCAB_SIZE)

element_to_index, index_to_element = construct_mapping(vocab)

########## Build the dataset ##########
'''
So given the corpus of data, we will build a dataset: X -> Y mapping 
Our model will learn to predict the next character given a context 'CONTEXT_LENGTH' characters long 
We will divide the data into training, dev, test sets.
'''
Xtr, Ytr, Xdev, Ydev, Xte, Yte = split(corpus, index_to_element, element_to_index)

########## Build the model ##########
embedding_dim = EMBEDDING_DIM
hidden_dim = HIDDEN_DIM
context_length = CONTEXT_LENGTH
parameters, bnmean_running, bnstd_running = gen_model_params(embedding_dim=embedding_dim, hidden_dim=hidden_dim, context_length=context_length, vocab_size=VOCAB_SIZE)

if MINOR_DEBUG:
    print([p.nelement() for p in parameters])
    print(bnmean_running.nelement(), bnstd_running.nelement())
######### Enable Gradients #########
parameters = enable_gradients(parameters)

## Training Loop 
steps, losses, bnmean_running, bnstd_running = training_loop(parameters, bnmean_running, bnstd_running, batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE, Xtr=Xtr, Ytr=Ytr)
plot_loss(steps, losses)

## Evaluate the model 
_, test_loss_value = evaluate_model(parameters, 'test', Xtr, Ytr, Xdev, Ydev, Xte, Yte, batch_norm=True, bnmean_running=bnmean_running, bnstd_running=bnstd_running)
print(f"Test Loss: {test_loss_value}")

_, dev_loss_value = evaluate_model(parameters, 'dev', Xtr, Ytr, Xdev, Ydev, Xte, Yte, batch_norm=True, bnmean_running=bnmean_running, bnstd_running=bnstd_running)
print(f"Dev Loss: {dev_loss_value}")

## Sample from the model 
sample_from_model(parameters, context_length, num_generation=30, index_to_element=index_to_element, 
                 batch_norm=True, bnmean_running=bnmean_running, bnstd_running=bnstd_running)