import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
lookup = {char: i+1 for i, char in enumerate(chars)}
lookup['.'] = 0
reverse_lookup = {i: char for char, i in lookup.items()}


x_chars, y_chars = [], []
for word in words:
    chars = ['.'] + list(word) + ['.']
    for char1, char2 in zip(chars, chars[1:]):
        i1 = lookup[char1]
        i2 = lookup[char2]
        x_chars.append(i1)
        y_chars.append(i2)

x_tensor = torch.tensor(x_chars)
y_tensor = torch.tensor(y_chars)
num = x_tensor.nelement()
x_encoded = F.one_hot(x_tensor, num_classes=27).float()

generator = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=generator, requires_grad=True)

for i in range(200):
    logits = x_encoded @ W
    counts = logits.exp()
    prob_dist = counts / counts.sum(1, keepdim=True)
    loss = -prob_dist[torch.arange(num), y_tensor].log().mean() + 0.01 * (W**2).mean()

    W.grad = None
    loss.backward()
    W.data -= 50 * W.grad

    print(loss.item())

for i in range(10):
    out = []
    ix = 0
    while True:
        x_encoded = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = x_encoded @ W
        counts = logits.exp()
        prob_dist = counts / counts.sum(1, keepdim=True)
        ix = torch.multinomial(prob_dist, num_samples=1, replacement=True, generator=generator).item()
        out.append(reverse_lookup[ix])
        if ix == 0:
            break
    print(''.join(out))