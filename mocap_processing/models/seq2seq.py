import torch
import torch.nn as nn
import random


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, input):
        _, (lstm_hidden, lstm_cell) = self.lstm(input)
        return lstm_hidden, lstm_cell


class DecoderStep(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DecoderStep, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden=None, cell=None):
        if (hidden is None) and (cell is None):
            output, (hidden, cell) = self.lstm(input)
        else:
            output, (hidden, cell) = self.lstm(input, (hidden, cell))
        output = output.squeeze(0)
        output = self.out(output)
        return output, hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, device='cuda'):
        super(LSTMDecoder, self).__init__()
        self.decoder = DecoderStep(input_dim, output_dim, hidden_dim)
        self.device = device

    def forward(self, trg, hidden=None, cell=None, teacher_forcing_ratio=0.5):
        trg = trg.transpose(0, 1)
        max_len = trg.shape[0]
        batch_size = trg.shape[1]

        input = trg[0, :]
        outputs = torch.zeros(max_len, batch_size, self.decoder.input_dim).to(self.device)
        for t in range(max_len):
            input = input.unsqueeze(0)
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[t] if teacher_force else output

        outputs = outputs.transpose(0, 1)
        return outputs


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        hidden, cell = self.encoder(src)
        outputs = self.decoder(trg, hidden, cell)
        return outputs