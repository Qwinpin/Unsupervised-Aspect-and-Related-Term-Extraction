import numpy as np
import torch
from torch import nn


def batch_generator(texts, iterations=1, batch_size=32, device=None):
    np.random.shuffle(texts)
    all_batches_number = len(texts) // batch_size
    if iterations == 0:
        iterations = all_batches_number
    print('Process {} iterations. {} available.'.format(iterations, all_batches_number))
    for _ in range(iterations):
        resulted_batch = []

        for _ in range(batch_size):
            anchor_idx = torch.tensor(np.random.choice(len(texts), size=1)[0])
            anchor_text = texts[anchor_idx]

            resulted_batch.append(torch.tensor(anchor_text))

        if device is not None:
            yield torch.tensor(nn.utils.rnn.pad_sequence(resulted_batch)).long().transpose(0, 1).to(device)
        else:
            yield torch.tensor(nn.utils.rnn.pad_sequence(resulted_batch)).long().transpose(0, 1)


def neg_batch_generator(texts, neg_size=10, batch_size=32, device=None):
    while True:
        resulted_batch = []
        for _ in range(batch_size):
            anchor_idx = torch.tensor(np.random.choice(len(texts), size=neg_size))
            anchor_text = texts[anchor_idx]

            resulted_batch.append(torch.tensor(anchor_text))

        if device is not None:
            yield torch.tensor(nn.utils.rnn.pad_sequence(resulted_batch)).long().transpose(0, 1).to(device)
        else:
            yield torch.tensor(nn.utils.rnn.pad_sequence(resulted_batch)).long().transpose(0, 1)