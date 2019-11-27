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
  def __init__(self, feature_list, model_path, index_path):
    self.model_path = model_path
    self.feature_list = feature_list
    self.index_path = index_path
    self.config = get_config(mode='train')
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.weights = torch.load(self.model_path)
    self.model = AutoLSTM(self.config.input_size, self.config.hidden_size,
                          self.config.num_layers, bidirectional=False)
    self.model.to(self.device)
    self.model.elstm.load_state_dict(self.weights)
    self.model.eval()
    self.data_parser = DatasetParser(self.feature_list)

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
    with open(self.index_path) as fp:
      pickle.dump(results, fp)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Build index for encoded video features')

  parser.add_argument('-l', '--features_list', dest='features_list',
                      help='Text file of video features list (.npy files)')
  parser.add_argument('-d', '--destination', dest='destination',
                      help='Location where to store the features index')
  parser.add_argument('-m', '--model_path', dest='model_path',
                      help='Lstm autoencoder model path')

  args = parser.parse_args()
  index = index(args.features_list, args.model_path, args.destination)
  index.build_feature_index()
