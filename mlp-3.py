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

random.seed(10)
random.shuffle(words)
num_words = len(words)
n1 = int(0.8*num_words)
n2 = int(0.9*num_words)

X_train, Y_train = build_dataset(words[:n1])
X_val, Y_val = build_dataset(words[n1:n2])
X_test, Y_test = build_dataset(words[n2:])

num_chars = len(lookup)
emb_size = 10
num_neurons = 200

g = torch.Generator().manual_seed(10)
C = torch.randn((num_chars, emb_size), generator=g)
W1 = torch.randn((emb_size * block_size, num_neurons), generator=g) * 5/3 / (emb_size * block_size) ** 0.5 
W2 = torch.randn((num_neurons, num_chars), generator=g) * 0.01
b2 = torch.randn(num_chars, generator=g) * 0
bngain = torch.ones((1, num_neurons))
bnbias = torch.zeros((1, num_neurons))
bnmean_running = torch.zeros((1, num_neurons))
bnstd_running = torch.zeros((1, num_neurons))

parameters = [C, W1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True

lossi = []
max_steps = int(2e5)
batch_size = 128
lr = 0.1
for i in range(max_steps):

    ix = torch.randint(0, X_train.shape[0], (batch_size,))
    X_batch, Y_batch = X_train[ix], Y_train[ix]

    emb = C[X_batch]
    emb_cat = emb.view(-1, emb_size * block_size)
    h_preact = emb_cat @ W1
    bnmeani = h_preact.mean(0, keepdim=True)
    bnstdi = h_preact.std(0, keepdim=True)
    h_preact = bngain * (h_preact - bnmeani) / bnstdi + bnbias

    with torch.no_grad():
        bnmean_running = bnmean_running * 0.999 + bnmeani * 0.001
        bnstd_running = bnstd_running * 0.999 + bnstdi * 0.001

    h = torch.tanh(h_preact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y_batch)

    for p in parameters:
        p.grad = None
    loss.backward()

    if i % max_steps/2 == 0:
        lr /= 10
    for p in parameters:
        p.data -= lr * p.grad

    lossi.append(loss.item())

@torch.no_grad()
def split_loss(data_split):
    x, y = {
        'training': (X_train, Y_train),
        'validation': (X_val, Y_val),
        'test': (X_test, Y_test)
    }[data_split]

    emb = C[x]
    emb_cat = emb.view(-1, emb_size * block_size)
    h_preact = emb_cat @ W1
    h_preact = bngain * (h_preact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(h_preact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(f'{data_split} loss: {loss.item()}')

split_loss('training')
split_loss('validation')
split_loss('test')

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        emb_cat = emb.view(-1, emb_size * block_size)
        h_preact = emb_cat @ W1
        h_preact = bngain * (h_preact - bnmean_running) / bnstd_running + bnbias
        h = torch.tanh(h_preact)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(reverse_lookup[i].replace('.', '') for i in out))

plt.plot(lossi)
plt.show()