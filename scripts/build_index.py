import argparse
import pickle

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from auto_encoder.lstm_network import AutoLSTM
from configs.config import get_config
from tools.dataset import DatasetParser
from tools.utils import my_collate


class index(object):
  def __init__(self):
    self.config = get_config()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.weights = torch.load(self.config.model_path)
    self.model = AutoLSTM(self.config.input_size, self.config.hidden_size,
                          self.config.num_layers, bidirectional=False)
    self.model.to(self.device)
    self.model.elstm.load_state_dict(self.weights)
    self.model.eval()
    self.data_parser = DatasetParser(self.config.features_list)
    self.dl = DataLoader(self.data_parser, batch_size=self.config.batch_size,
                         shuffle=False,
                         num_workers=12, collate_fn=my_collate)
    self.pbar = tqdm(self.dl)

  def build_feature_index(self):
    results = {}
    for b in self.pbar:
      (filepath, inputs, inputs_rev) = b
      inputs = inputs.to(self.device)
      outputs, (h_n, c_n) = self.model.elstm(inputs)
      outputs = outputs.detach().cpu().numpy()
      res = dict(zip(filepath, outputs))
      results.update(res)
    with open(self.config.index_path) as fp:
      pickle.dump(results, fp)


if __name__ == '__main__':
  index = index()
  index.build_feature_index()
