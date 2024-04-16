# See Karpathy's shared google colab jupyter notebook

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

print(text[:1000])

# get the set of all the characters that occur in text set(text)
# turn that into an arbitrary ordering list()
# sort the ordering sorted
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)
# possible elements characters the model can see or output

# TOKENIZE: convert the raw text as a string
# into some sequence of integers
# according to some vocabulary of possibl elements
# We will just make a character level language model
# translate individual characters into integers

# create a mapping from characters to integers
char_to_int_map = { ch:i for i,ch in enumerate(chars) }
int_to_char_map = { i:ch for i,ch in enumerate(chars) }
# encoder: take a string, output a list of integers
encode = lambda s:[char_to_int_map[c] for c in s]
# decoder: take a list of integers, output a string
decode = lambda l: ''.join([int_to_char_map[i] for i in l])

print(encode("HELLO WORLD ARJUN HERE!"))
print(decode(encode("HELLO WORLD ARJUN HERE!")))

# TOKENIZATION: translation to integers and back for an arbitrary string
# done on a character level, usually done on more complex tokens
# many other schemas for tokenizaTION (RATHER THAN CHAR LEVEL)
# google sentencepiece
# subwords -> not encoding entire words, but also not individual characters
# tiktoken -> tokenizer, has 50,000 tokens (instead of 65 for our character level based)
# trade-off codebook size, sequence lengths
# the more tokens in vocab, the smaller the integer sequence translation of a string is
# TOKENIZATION: a way to break up any arbitrary string into a list of integers (recognized tokens in vocab)

# encode the entire text dataset and store it into a pytorch Tensor
import torch
# take all of the text in tiny shakespeare, encode it
# and then wrap it in torch.tensor
# to get the data tensor
data = torch.tensor(encode(text))
# print(data.shape, data.dtype)
# print(data[:1000])
# this is the first 1000 characters from above in the character level tokens

# separate data into train and validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# train transformer on chunks of data (blocks) at a time instead of all at once
# makes it computationally tractable
block_size = 8 # context length
print(train_data[:block_size+1])

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences we process in a single pass in parallel
block_size = 8 # maximum context length for predictions?

def get_batch(split):
    data = train_data if split =="train" else val_data
    # start at random places in training set, generate one start location for each batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # first block size characters starting at i
    # chunks for every one of i in ix, gget all the batches
    x = torch.stack([data[i:i+block_size] for i in ix])
    # are the offset by one of that
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # use torch.stack to stack them up into rows
    # they all become rows in a 4 by 8 tensor
    # 4 blocks in each batch of training set
    # each block has 8 character long text chunks
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)

print('targets:')
print(yb.shape)
print(yb)

print('----')

# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, :t+1]
#         target = yb[b, t]
#         print(f"when input is {context.tolist()} the target: {target}")

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # logits -> scores for next character in the sequence
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # stretch out all the batches and make channel second dim
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # how well are we predicting the targets when comapred to logits, predictions
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predcitions
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

# batch is just one, time is just one, holds a zeroes
# 0 -> newline character is the first in the sequence
idx = torch.zeros((1, 1), dtype=torch.long)
# [0] unplucks batch
generated_tokens = m.generate(idx, max_new_tokens=100)[0].tolist()
print(decode(generated_tokens))
