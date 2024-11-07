import torch
from random import randint

class TimeSeriesDataset():

    def __init__(self, data, external_inputs=None, sequence_length=200, batch_size=16):
        self.X = torch.tensor(data, dtype=torch.float32)
        self.total_time_steps = self.X.shape[0]
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        if external_inputs is not None:
            self.S = torch.tensor(external_inputs, dtype=torch.float32)
            assert self.X.size(0) == self.S.size(0), "X and S must have the same number of time steps"
        else:
            self.S = None

    def __len__(self):
        return self.total_time_steps - self.sequence_length - 1

    def __getitem__(self, t):

        x = self.X[t:t+self.sequence_length, :]
        y = self.X[t+1:t+self.sequence_length+1, :]
        if self.S is None:
            return x, y, None
        else:
            s = self.S[t:t+self.sequence_length, :]
            return x, y, s

    def sample_batch(self):
        """
        Sample a batch of sequences.
        """
        X = []
        Y = []
        S = []
        for _ in range(self.batch_size):
            idx = randint(0, len(self))
            x, y, s = self[idx]
            X.append(x)
            Y.append(y)
            S.append(s)

        if S[0] is None:
            return torch.stack(X), torch.stack(Y), None
        else:
            return torch.stack(X), torch.stack(Y), torch.stack(S)