import tkinter as tk
from tkinter import filedialog as fd
import torch
import torch.nn as nn
from torch.nn import functional as F

def open_text_file():
    # Open a file dialog to select a text file
    file_path = fd.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        # Read the contents of the file
        with open(file_path, 'r') as file:
            content = file.read()
        # Print the contents to the terminal
        chars = sorted(list(set(content)))
        vocabSize = len(chars)
        print(''.join(chars))
        print(vocabSize)
    return chars, content
        
chars, content = open_text_file()
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hii there"))
print(decode(encode("hii there")))

data = torch.tensor(encode(content), dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")
    
torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data))
    x = torch.stack([data[i:i+block_size], for i in ix])
        

# Initialize the Tkinter root window
root = tk.Tk()
root.withdraw()  # Hide the root window