import os
import random

import numpy as np
from torch.utils.data import Dataset

from configs.config import get_config
from tools.utils import get_logger
from tools.utils import read_filelist

logger = get_logger(__name__)


class DatasetParser(Dataset):

  def __init__(self, feautures_list):

    try:
      self.config = get_config(mode='train')
      self.features_list = feautures_list
      self.batch_size = self.config.nb_samples
      self.min_ts = self.config.min_timesteps
      self.max_ts = self.config.max_timesteps
      self.input_features_dim = self.config.input_size

      self.batcher = None
      self.feature_files = read_filelist(self.features_list)

      assert len(self.feature_files) != 0, 'Empty feature'
      'file list at {}'.format(self.features_list)

      self.batcher = self.batch_gen()

      logger.info('Setting DatasetParser from {}'.format(self.features_list))

    except Exception as err:
      logger.warning('Error setting up DatasetParser {}'.format(err))

  def fetch_batch(self):

    file_batch = next(self.batcher, None)

    if file_batch is None:
      logger.info('Resetting batch gen EOF reached')
      self.batcher = self.batch_gen()
      file_batch = next(self.batcher)

    file_path, feature_batch, feature_batch_rev, mask_rev = self.read_features(
      file_batch)
    return file_path, file_batch, feature_batch, feature_batch_rev, mask_rev

  def read_feature(self, filepath):

    f = np.load(
      os.path.join('/nas/team-space/experiments/drive-features', filepath))
    # sample a window from all timesteps
    # of atleast self.min_ts and atmost self.max_ts
    if f.shape[0] < self.min_ts:
      return None

    ts_begin = random.randint(0, f.shape[0] - self.min_ts)
    ts_end = random.randint(ts_begin + self.min_ts,
                            min(ts_begin + self.max_ts, f.shape[0]))

    # get the features from that window
    temp_win = np.zeros([self.max_ts, self.input_features_dim])
    temp_win_rev = np.zeros_like(temp_win)
    mask_rev = np.zeros([self.max_ts])
    mask_rev[0: ts_end - ts_begin] = 1
    temp_win[0: ts_end - ts_begin, :] = f[ts_begin:ts_end, :]
    temp_win_rev[0: ts_end - ts_begin, :] = f[ts_begin:ts_end, :][::-1, :]

    return filepath, temp_win, temp_win_rev, mask_rev

  def read_features(self, batch_list):

    features = []
    features_rev = []
    features_rev_mask = []

    for idx, filepath in enumerate(batch_list):

      try:

        filepath, temp_win, temp_win_rev, mask_rev = self.read_feature(filepath)

        features.append(temp_win)
        features_rev.append(temp_win_rev)
        features_rev_mask.append(mask_rev)

      except Exception as err:
        logger.warning('Error parsing {}, {}'.format(filepath, err))

    return filepath, features, features_rev, features_rev_mask

  def batch_gen(self):

    random.shuffle(self.feature_files)

    for i in range(0, len(self.feature_files), self.batch_size):
      yield self.feature_files[i:i + self.batch_size]

  def __getitem__(self, index):

    d = self.read_feature(self.feature_files[index])

    if d is None:
      return None
    (filepath, feat, feat_rev, _) = d
    feat = feat.astype(np.float32)
    feat_rev = feat_rev.astype(np.float32)

    return filepath, feat, feat_rev

  def __iter__(self):

    index = random.randint(0, len(self.feature_files))

    d = self.read_feature(self.feature_files[index])
    if d is None:
      return None
    (filepath, feat, feat_rev, _) = d
    feat = feat.astype(np.float32)
    feat_rev = feat_rev.astype(np.float32)

    return feat, feat_rev

  def __len__(self):

    return len(self.feature_files)
