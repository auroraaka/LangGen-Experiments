import torch
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
lookup = {char: i+1 for i, char in enumerate(chars)}
lookup['.'] = 0
reverse_lookup = {i: char for char, i in lookup.items()}

for word in words:
    chars = ['.'] + list(word) + ['.']
    for char1, char2 in zip(chars, chars[1:]):
        i1 = lookup[char1]
        i2 = lookup[char2]
        N[i1, i2] += 1

generator = torch.Generator().manual_seed(10)

P = (N+1).float()
P /= P.sum(1, keepdim=True)

for i in range(10):
    out = []
    ix = 0
    while True:
        prob_dist = P[ix]
        ix = torch.multinomial(prob_dist, num_samples=1, replacement=True, generator=generator).item()
        out.append(reverse_lookup[ix])
        if ix == 0:
            break
    # print(''.join(out))


log_likelihood = 0.0
n = 0
for word in words:
    chars = ['.'] + list(word) + ['.']
    for char1, char2 in zip(chars, chars[1:]):
        i1 = lookup[char1]
        i2 = lookup[char2]
        log_likelihood += torch.log(P[i1, i2])
        n += 1
nll = -log_likelihood
print(f'Loss: {nll/n}')