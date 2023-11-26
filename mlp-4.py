from typing import Any
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

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in ** (1/2)
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.001):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []


num_emb = 10
num_hidden = 100
num_chars = len(lookup)
g = torch.Generator().manual_seed(10)

C = torch.randn((num_chars, num_emb))
layers = [
  Linear(num_emb * block_size, num_hidden, bias=False), Tanh(),
  Linear(num_hidden, num_hidden, bias=False), Tanh(),
  Linear(num_hidden, num_hidden, bias=False), Tanh(),
  Linear(num_hidden, num_hidden, bias=False), Tanh(),
  Linear(num_hidden, num_hidden, bias=False), Tanh(),
  Linear(num_hidden, num_chars, bias=False),
]

with torch.no_grad():
    layers[-1].weight *= 0.1
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
for p in parameters:
    p.requires_grad = True

max_steps = int(2e5)
batch_size = 32
lossi = []
udr = []
lr = 0.1
for i in range(max_steps):

    ix = torch.randint(0, X_train.shape[0], (batch_size,), generator=g)
    X_batch, Y_batch = X_train[ix], Y_train[ix]

    emb = C[X_batch]
    x = emb.view(-1, num_emb * block_size)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Y_batch)

    for p in parameters:
        p.grad = None
    loss.backward()

    if i % max_steps/2 == 0:
        lr /= 10
    for p in parameters:
        p.data -= lr * p.grad

    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    with torch.no_grad():
        udr.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])


@torch.no_grad()
def split_loss(data_split):
    x, y = {
        'training': (X_train, Y_train),
        'validation': (X_val, Y_val),
        'test': (X_test, Y_test)
    }[data_split]

    emb = C[x]
    x = emb.view(-1, num_emb * block_size)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, y)
    print(f'{data_split} loss: {loss.item()}')

split_loss('training')
split_loss('validation')
split_loss('test')

g = torch.Generator().manual_seed(12)
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        x = emb.view(emb.shape[0], -1)
        for layer in layers:
            x = layer(x)
        probs = F.softmax(x, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    
    print(''.join(reverse_lookup[i].replace('.', '') for i in out))

def plot_histogram(data, legends, title):
    plt.figure(figsize=(20, 4))
    for d, l in zip(data, legends):
        plt.plot(d[0], d[1])
    plt.legend(legends)
    plt.title(title)

data = []
legends = []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out
        # print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean() * 100))
        hy, hx = torch.histogram(t, density=True)
        data.append((hx[:-1].detach(), hy.detach()))
        legends.append(f'layer {i} {layer.__class__.__name__}')
plot_histogram(data, legends, 'Activation Distribution')

data = []
legends = []
for i, p in enumerate(parameters):
    t = p.grad
    if p.ndim == 2:
        # print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
        hy, hx = torch.histogram(t, density=True)
        data.append((hx[:-1].detach(), hy.detach()))
        legends.append(f'{i} {tuple(p.shape)}')
plot_histogram(data, legends, 'Weights Gradient Distribution')

plt.figure(figsize=(20, 4))
legends = []
for i,p in enumerate(parameters):
  if p.ndim == 2:
    plt.plot([udr[j][i] for j in range(len(udr))])
    legends.append('param %d' % i)
plt.plot([0, len(udr)], [-3, -3], 'k')
plt.legend(legends)
plt.title('Update to Data Ratio')

plt.show()

