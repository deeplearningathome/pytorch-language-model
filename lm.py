import torch.nn as nn
from torch.autograd import Variable

class LM_LSTM(nn.Module):
  """Simple LSMT-based language model"""
  def __init__(self, embedding_dim, num_steps, batch_size, vocab_size, num_layers, dp_keep_prob):
    super(LM_LSTM, self).__init__()
    self.embedding_dim = embedding_dim
    self.num_steps = num_steps
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.dp_keep_prob = dp_keep_prob
    self.num_layers = num_layers
    self.dropout = nn.Dropout(1 - dp_keep_prob)
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=embedding_dim,
                            num_layers=num_layers,
                            dropout=1 - dp_keep_prob)
    self.sm_fc = nn.Linear(in_features=embedding_dim,
                           out_features=vocab_size)
    self.init_weights()

  def init_weights(self):
    init_range = 0.1
    self.word_embeddings.weight.data.uniform_(-init_range, init_range)
    self.sm_fc.bias.data.fill_(0.0)
    self.sm_fc.weight.data.uniform_(-init_range, init_range)

  def init_hidden(self):
    weight = next(self.parameters()).data
    return (Variable(weight.new(self.num_layers, self.batch_size, self.embedding_dim).zero_()),
            Variable(weight.new(self.num_layers, self.batch_size, self.embedding_dim).zero_()))

  def forward(self, inputs, hidden):
    embeds = self.dropout(self.word_embeddings(inputs))
    lstm_out, hidden = self.lstm(embeds, hidden)
    lstm_out = self.dropout(lstm_out)
    logits = self.sm_fc(lstm_out.view(-1, self.embedding_dim))
    return logits.view(self.num_steps, self.batch_size, self.vocab_size), hidden

def repackage_hidden(h):
  """Wraps hidden states in new Variables, to detach them from their history."""
  if type(h) == Variable:
    return Variable(h.data)
  else:
    return tuple(repackage_hidden(v) for v in h)