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

import wget
import os
from typing import List

# Context length will determine, how many characters will be used to predict the next character
CONTEXT_LENGTH  = 3 

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

########## Get the data #########
corpus = get_data("https://raw.githubusercontent.com/karpathy/makemore/master/names.txt", "names.txt")

# print(f"Corpus length: {len(corpus)}")
# print(f"first 10 names: {corpus[:10]}")

######### Construct a vocabulary for the model using the corpus #########
one_big_blob = ''.join(corpus)
unique_chars = set(one_big_blob)
list_unique_chars = list(unique_chars)
vocab = sorted(list_unique_chars)
# construct a mapping from each element in vocab to an index 
element_to_index = {element:index+1 for index,element in enumerate(vocab)}
# add a padding element
element_to_index['.'] = 0 
# index_to_element 
index_to_element = {value:key for key,value in element_to_index.items()}

# print(index_to_element)

