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
CONTEXT_LENGTH  = 3
BATCH_SIZE = 32
EPOCHS = 200000
LEARNING_RATE = 0.1
HIDDEN_DIM = 200
EMBEDDING_DIM = 10
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

def gen_model_params(embedding_dim: int, hidden_dim: int, context_length: int):
    '''
    Model Architecture: A Neural Probabilistic Language Model 
    Authors: Yoshua Bengio et al
    '''
    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((27, embedding_dim), generator=g)
    W1  = torch.randn((embedding_dim*context_length, hidden_dim), generator=g)
    b1 = torch.randn((hidden_dim), generator=g)
    W2 = torch.randn((hidden_dim, 27), generator=g)
    b2 = torch.randn((27), generator=g)
    parameters = [C, W1, b1, W2, b2]
    return parameters

def enable_gradients(parameters):
    for p in parameters:
        p.requires_grad = True
    return parameters

def training_loop(parameters, batch_size, epochs, learning_rate, Xtr, Ytr):
    C, W1, b1, W2, b2 = parameters
    steps = []
    losses = []
    for i in range(epochs):
        # construct the batch 
        index = torch.randint(0, Xtr.shape[0], (batch_size,))
        
        # text embedding 
        embedding = C[Xtr[index]] # Embedding matrix is a lookup table for the characters; which we learn from the data 
        
        # forward pass 
        h = torch.tanh(embedding.view(batch_size, -1) @ W1 + b1)
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
        
    return steps, losses
    
def evaluate_model(parameters, Xte, Yte):
    C, W1, b1, W2, b2 = parameters
    embedding = C[Xte]
    h = torch.tanh(embedding.view(Xte.shape[0], -1) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yte)
    return loss.item()

def plot_loss(steps, losses):
    plt.plot(steps, losses)
    plt.savefig('loss_plot.png')

def sample_from_model(parameters, context_length, num_generation, index_to_element):
    C, W1, b1, W2, b2 = parameters

    for _ in range(num_generation):
        generated_text = []
        context = [0] * context_length
        while True:
            embedding = C[torch.tensor([context])]
            h = torch.tanh(embedding.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            next_char_index = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [next_char_index]
            generated_text.append(next_char_index)
            if next_char_index == 0:
                break
        print(''.join(index_to_element[i] for i in generated_text))

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

########## Build the model ##########
embedding_dim = EMBEDDING_DIM
hidden_dim = HIDDEN_DIM
context_length = CONTEXT_LENGTH
parameters = gen_model_params(embedding_dim=embedding_dim, hidden_dim=hidden_dim, context_length=context_length)

if MINOR_DEBUG:
    print(p.nelement() for p in parameters)

######### Enable Gradients #########
parameters = enable_gradients(parameters)

## Training Loop 
steps, losses = training_loop(parameters, batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE, Xtr=Xtr, Ytr=Ytr)
plot_loss(steps, losses)

## Evaluate the model 
loss = evaluate_model(parameters, Xte, Yte)
print(f"Test Loss: {loss}")

## Sample from the model 
sample_from_model(parameters, context_length, num_generation=30, index_to_element=index_to_element)