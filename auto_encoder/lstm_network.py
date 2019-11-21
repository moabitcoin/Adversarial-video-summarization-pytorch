import torch
import torch.nn as nn


class eLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers=2,
               bidirectional=False):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bidirectional = bidirectional
    # hidden_size should be 1024
    self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                        bidirectional=bidirectional)

  """
  Args:
      x: weighted_features, (seq_len, 1, input_size) = (seq_len, 1, 2048)
  Return:
      fixed length feature: a tuple of hidden state, each of (num_layers * 1,
      1, hidden_size) = (2, 1, 1024)
  """

  def forward(self, x):
    self.lstm.flatten_parameters()
    out, (h_n, c_n) = self.lstm(x)

    return out, (h_n, c_n)


# decoder,
class dLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers,
               bidirectional=False):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bidirectional = bidirectional
    self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                        bidirectional=self.bidirectional)

  """
  Args:
      x: fixed length feature, (seq_len, 1, input_size) = (seq_len, 1, 4096)
  Return:
      decoded: decoded feature of a video, (seq_len, 1, hidden_size) = (
      seq_len, 1, 2048)
  """

  def forward(self, input, state_inits):
    self.lstm.flatten_parameters()
    return self.lstm(input, state_inits)


# consist of eLSTM and dLSTM
class AutoLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers,
               bidirectional=False):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bidirectional = bidirectional
    self.elstm = eLSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional)
    if bidirectional:
      self.dlstm = dLSTM(2 * self.hidden_size, self.input_size, self.num_layers)
    else:
      self.dlstm = dLSTM(self.hidden_size, self.input_size, self.num_layers)
    self.linear = nn.Linear((self.input_size // self.hidden_size) * self.hidden_size, self.input_size)
  """
  Args:
      x: weighted_features, (seq_len, 1, input_size) = (seq_len, 1, 2048)
  Return:
      decoded: decoded feature of a video, (seq_len, 1, 2 * hidden_size) = (
      seq_len, 1, 2048)
  """

  def forward(self, x):
    encoded_t, (h_n, c_n) = self.elstm(x)
    sequence_length, batch_size, feat_size = encoded_t.size()
    # getting output at the last time step
    encoded = encoded_t[-1].reshape([1, batch_size, feat_size])
    # tile/repeat it seq_length time
    encoded = encoded.repeat([sequence_length, 1, 1])
    h0_encoder_bi = torch.zeros(self.num_layers, batch_size,
                                self.input_size).cuda()
    c0_encoder_bi = torch.zeros(self.num_layers, batch_size,
                                self.input_size).cuda()
    decoded, _ = self.dlstm(encoded, (h0_encoder_bi, c0_encoder_bi))

    y = self.linear(decoded)

    return y
