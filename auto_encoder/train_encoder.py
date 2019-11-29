import copy
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from auto_encoder.lstm_network import AutoLSTM
from configs.config import get_config
from tools.dataset import DatasetParser
from tools.tf_log_writer import LogWriter
from tools.utils import my_collate


class trainer(object):
  def __init__(self):
    self.config = get_config()
    self.log_writer = LogWriter(self.config.log_dir)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
      print("Using CUDA, benchmarking implementations", file=sys.stderr)
      torch.backends.cudnn.benchmark = True
    self.model = AutoLSTM(self.config.input_size, self.config.hidden_size,
                          self.config.num_layers, bidirectional=False)
    self.model.to(self.device)
    self.model = nn.DataParallel(self.model)

  def train_model(self):
    step = 0
    losses = []
    since = time.time()
    criterion = nn.MSELoss()
    train_parser = DatasetParser(self.config.features_list)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    best_model_wts = copy.deepcopy(self.model.state_dict())
    dl = DataLoader(train_parser, batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=self.config.num_workers, collate_fn=my_collate)
    for epoch in range(self.config.n_epochs):
      train_loss = 0
      pbar = tqdm(dl, desc='Batch', ascii=True)
      try:
        for b in pbar:
          (filepath, inputs, inputs_rev) = b
          inputs = inputs.float().transpose(0, 1)
          inputs_rev = inputs_rev.float().transpose(0, 1)

          inputs = inputs.to(self.device)
          inputs_rev = inputs_rev.to(self.device)

          optimizer.zero_grad()

          outputs = self.model(inputs)

          loss = criterion(outputs, inputs_rev)
          # loss = criterion(outputs, inputs)
          loss.backward()
          optimizer.step()
          losses.append(loss)

          train_loss += loss.item() * inputs.size(1)
          step += 1

          pbar.set_description('loss {}'.format(loss.item()))

          if step % self.config.save_interval == 0:
            torch.save(self.model.module.state_dict(),
                       '{}/autoencoder_encoder-3l-{}.pth'.format(
                         self.config.model_save_dir, step))
        au_loss = torch.stack(losses).mean()
        self.log_writer.update_loss(au_loss, epoch,
                                    'autoencoder loss: False')
      except Exception as e:
        print(e)
        continue
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:0f}s'.format(time_elapsed // 60,
                                                       time_elapsed % 60))

    self.model.load_state_dict(best_model_wts)
    return self.model, losses
if __name__ == '__main__':
  t = trainer()
  t.train_model()
