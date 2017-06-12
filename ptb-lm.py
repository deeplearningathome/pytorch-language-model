import argparse
import time
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
from lm import repackage_hidden, LM_LSTM
import reader
import numpy as np

parser = argparse.ArgumentParser(description='Simplest LSTM-based language model in PyTorch')
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus')
parser.add_argument('--hidden_size', type=int, default=1500,
                    help='size of word embeddings')
parser.add_argument('--num_steps', type=int, default=35,
                    help='number of LSTM steps')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of LSTM layers')
parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs')
parser.add_argument('--dp_keep_prob', type=float, default=0.35,
                    help='dropout *keep* probability')
parser.add_argument('--inital_lr', type=float, default=20.0,
                    help='initial learning rate')
parser.add_argument('--save', type=str,  default='lm_model.pt',
                    help='path to save the final model')
args = parser.parse_args()

criterion = nn.CrossEntropyLoss()
def run_epoch(model, data, is_train=False, lr=1.0):
  """Runs the model on the given data."""
  if is_train:
    model.train()
  else:
    model.eval()
  epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
  start_time = time.time()
  hidden = model.init_hidden()
  costs = 0.0
  iters = 0
  for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.num_steps)):
    inputs = Variable(torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous()).cuda()
    model.zero_grad()
    hidden = repackage_hidden(hidden)
    outputs, hidden = model(inputs, hidden)
    targets = Variable(torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous()).cuda()
    tt = torch.squeeze(targets.view(-1, model.batch_size * model.num_steps))

    loss = criterion(outputs.view(-1, model.vocab_size), tt)
    costs += loss.data[0] * model.num_steps
    iters += model.num_steps

    if is_train:
      loss.backward()
      torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
      for p in model.parameters():
        p.data.add_(-lr, p.grad.data)
      if step % (epoch_size // 10) == 10:
        print("{} perplexity: {:8.2f} speed: {} wps".format(step * 1.0 / epoch_size, np.exp(costs / iters),
                                                       iters * model.batch_size / (time.time() - start_time)))
  return np.exp(costs / iters)


if __name__ == "__main__":
  raw_data = reader.ptb_raw_data(data_path=args.data)
  train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
  vocab_size = len(word_to_id)
  print('Vocabluary size: {}'.format(vocab_size))
  model = LM_LSTM(embedding_dim=args.hidden_size, num_steps=args.num_steps, batch_size=args.batch_size,
                  vocab_size=vocab_size, num_layers=args.num_layers, dp_keep_prob=args.dp_keep_prob)
  model.cuda()
  lr = args.inital_lr
  # decay factor for learning rate
  lr_decay_base = 1 / 1.15
  # we will not touch lr for the first m_flat_lr epochs
  m_flat_lr = 14.0

  print("########## Training ##########################")
  for epoch in range(args.num_epochs):
    lr_decay = lr_decay_base ** max(epoch - m_flat_lr, 0)
    lr = lr * lr_decay # decay lr if it is time
    train_p = run_epoch(model, train_data, True, lr)
    print('Train perplexity at epoch {}: {:8.2f}'.format(epoch, train_p))
    print('Validation perplexity at epoch {}: {:8.2f}'.format(epoch, run_epoch(model, valid_data)))
  print("########## Testing ##########################")
  model.batch_size = 1 # to make sure we process all the data
  print('Test Perplexity: {:8.2f}'.format(run_epoch(model, test_data)))
  with open(args.save, 'wb') as f:
    torch.save(model, f)
  print("########## Done! ##########################")