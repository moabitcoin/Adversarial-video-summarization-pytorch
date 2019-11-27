import argparse
import pickle

import numpy as np
import scipy
import torch
from scipy import spatial

from auto_encoder.lstm_network import AutoLSTM
from configs.config import get_config
from tools.utils import extract_video_features


class Matcher(object):
  def __init__(self):
    self.config = get_config()
    with open(self.config.index_path, 'rb') as fp:
      self.data = pickle.load(fp)
    self.file_names = []
    self.matrix = []
    for k, v in self.data.items():
      self.file_names.append(k)
      self.matrix.append(v)
    self.matrix = np.array(self.matrix)
    self.file_names = np.array(self.file_names)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.weights = torch.load(self.config.model_path)
    self.model = AutoLSTM(self.config.input_size, self.config.hidden_size,
                     self.config.num_layers, False)
    self.model.elstm.load_state_dict(self.weights)
    self.model.eval()
    self.model.to(self.device)

  def cos_cdist(self, vector):
    # getting cosine distance between video features query and video features
    # database
    v = vector.reshape(1, -1)
    res = []
    for q in range(0, len(self.matrix)):
      y = scipy.spatial.distance.cosine(self.matrix[q].reshape(1, -1), v)
      res.append(float(y))
    return res

  def match(self, features, topn=5):
    video_distances = self.cos_cdist(features)
    # getting top n videos
    res = dict(sorted(zip(self.file_names, video_distances)))
    firstnvals = {k: res[k] for k in sorted(res.keys())[:topn]}
    nearest_video_paths, best_distances = zip(*firstnvals.items())
    return nearest_video_paths, best_distances

  def search(self):
    d = extract_video_features(self.config.features_query)
    (filepath, inputs, inputs_rev, _) = d
    inputs = torch.from_numpy(inputs).float().unsqueeze(0).to(self.device)
    outputs, (h_n, c_n) = self.model.elstm(inputs)
    video_feat = outputs.cpu().detach().numpy()
    video_paths, cosine_distances = matcher.match(video_feat, self.config.topn)
    return video_paths, cosine_distances

if __name__ == '__main__':
  matcher = Matcher()
  a, p = matcher.search()
  print(a)
  print(p)
