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
