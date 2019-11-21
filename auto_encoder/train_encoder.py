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
from tools.utils import my_collate


class trainer(object):
  def __init__(self, config):
    from tools.tf_log_writer import LogWriter
    self.log_writer = LogWriter('/tmp/pycharm_project_753/logs')
    self.config = config
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
      print("Using CUDA, benchmarking implementations", file=sys.stderr)
      torch.backends.cudnn.benchmark = True

  def train_model(self, model, num_epoches, model_save_dir):

    step = 0
    losses = []
    num_workers = 12
    since = time.time()
    learning_rate = 1e-4

    model = model.to(self.device)
    model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    train_parser = DatasetParser(self.config.train_features_list)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_model_wts = copy.deepcopy(model.state_dict())
    dl = DataLoader(train_parser, batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=num_workers,collate_fn=my_collate)
    for epoch in range(num_epoches):
      train_loss = 0
      try:
        for (filepath, inputs, inputs_rev) in tqdm(dl):
          step +=1
          inputs = inputs.float().transpose(0, 1)
          inputs_rev = inputs_rev.float().transpose(0, 1)

          inputs = inputs.to(self.device)
          inputs_rev = inputs_rev.to(self.device)

          optimizer.zero_grad()

          outputs = model(inputs)

          loss = criterion(outputs, inputs_rev)
          # loss = criterion(outputs, inputs)
          loss.backward()
          optimizer.step()
          losses.append(loss)

          train_loss += loss.item() * inputs.size(1)
          print('loss {}'.format(loss.item()))
          torch.save(model.module.elstm.state_dict(),
                     '{}/encoder_lstm.pth'.format(model_save_dir))
          torch.save(model.module.state_dict(),
                     '{}/autoencoder_encoder.pth'.format(model_save_dir))
          au_loss = torch.stack(losses).mean()
          print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, au_loss))
          self.log_writer.update_loss(au_loss, epoch,
                                      'autoencoder loss')
      except TypeError :
        print('hna')
        print(type(filepath))
        print(type(inputs))
        print(type(inputs_rev))
        continue
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:0f}s'.format(time_elapsed // 60,
                                                       time_elapsed % 60))

    model.load_state_dict(best_model_wts)
    return model, losses


if __name__ == '__main__':
  num_epoches = 300

  config = get_config(mode='train')

  t = trainer(config)
  model = AutoLSTM(config.input_size, config.hidden_size,
                   config.num_layers, bidirectional=False)
  model.to(t.device)

  t.train_model(model, num_epoches, config.model_save_dir)
