import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
lookup = {char: i+1 for i, char in enumerate(chars)}
lookup['.'] = 0
reverse_lookup = {i: char for char, i in lookup.items()}

block_size = 3
def build_dataset(words):  
    X, Y = [], []
    for word in words:
        context = [0] * block_size
        for ch in word + '.':
            ix = lookup[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

random.seed(42)
random.shuffle(words)
num_words = len(words)
n1 = int(0.8*num_words)
n2 = int(0.9*num_words)

X_train, Y_train = build_dataset(words[:n1])
X_val, Y_val = build_dataset(words[n1:n2])
X_test, Y_test = build_dataset(words[n2:])

num_chars = len(lookup)
num_neurons = 200
emb_size = 10

g = torch.Generator().manual_seed(10)
C = torch.randn((num_chars, emb_size), generator=g)
W1 = torch.randn((emb_size * block_size, num_neurons), generator=g)
b1 = torch.randn(num_neurons, generator=g)
W2 = torch.randn((num_neurons, num_chars), generator=g)
b2 = torch.randn(num_chars, generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

stepi = []
lossi = []
batch_size = 128
lr = 0.1
for i in range(200000):

    ix = torch.randint(0, X_train.shape[0], (batch_size,))

    emb = C[X_train[ix]]
    h = torch.tanh(emb.view(-1, emb_size * block_size) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y_train[ix])

    for p in parameters:
        p.grad = None
    loss.backward()

    if i % 200000 == 0:
        lr /= 10

    for p in parameters:
        p.data -= lr * p.grad

    stepi.append(i)
    lossi.append(loss.item())

emb = C[X_train]
h = torch.tanh(emb.view(-1, emb_size * block_size) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y_train)
print(f'Training Loss: {loss.item()}')

emb = C[X_val]
h = torch.tanh(emb.view(-1, emb_size * block_size) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y_val)
print(f'Validation Loss: {loss.item()}')

emb = C[X_test]
h = torch.tanh(emb.view(-1, emb_size * block_size) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y_test)
print(f'Test Loss: {loss.item()}')

g = torch.Generator().manual_seed(10)

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
      emb = C[torch.tensor([context])]
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    print(''.join(reverse_lookup[i].replace('.', '') for i in out))

plt.plot(stepi, lossi)
plt.show()