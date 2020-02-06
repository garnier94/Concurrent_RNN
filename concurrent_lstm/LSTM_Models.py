import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_Model(nn.Module):
    def __init__(self, n_param, n_hidden, n_layer=1, dropout=0):
        super().__init__()
        self.rnn = nn.LSTM(n_param, n_hidden, n_layer,dropout=dropout)
        self.hidden_layer = (torch.zeros(n_layer, 1, n_hidden),
                             torch.zeros(n_layer, 1, n_hidden))
        self.linear = nn.Linear(n_hidden, 1)
        self.n_hidden = n_hidden
        self.n_layer = n_layer


    def forward(self, x):
        output, self.hidden_layer = self.rnn(x, self.hidden_layer)
        return F.softplus(self.linear(output))

    def reinitialize(self):
        self.hidden_layer = (torch.normal(0, 1, size=(self.n_layer, 1, self.n_hidden)),
                             torch.normal(0, 1, size=(self.n_layer, 1, self.n_hidden)))

    def predict_non_concurrent_step(self, step):
        self.reinitialize()
        return  self(step)
