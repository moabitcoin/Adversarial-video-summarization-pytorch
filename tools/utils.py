import logging
import sys
import numpy as np
import random
import yaml
from torch.utils.data.dataloader import default_collate

LOGGING_FORMAT = '[%(pathname)s][%(funcName)s:%(lineno)d]' + \
                 '[%(levelname)s] %(message)s'
LOGGING_STREAM = sys.stdout


def get_logger(logger_name, log_level=logging.DEBUG):
  logger = logging.getLogger(logger_name)
  logger.setLevel(log_level)
  ch = logging.StreamHandler(LOGGING_STREAM)
  formatter = logging.Formatter(LOGGING_FORMAT)
  ch.setFormatter(formatter)
  ch.setLevel(log_level)
  logger.addHandler(ch)
  logger.propagate = False

  return logger


logger = get_logger(__name__)


def read_yaml_file(filepath):
  try:
    with open(filepath, 'r') as pfile:
      content = yaml.load(pfile, Loader=yaml.FullLoader)

    return content

  except Exception as err:
    logger.error('Error reading yaml file {}'.format(filepath, err))


def read_filelist(filepath):
  try:
    with open(filepath, 'r') as pfile:
      content = pfile.readlines()
      content = [c.strip() for c in content]

    return content

  except Exception as err:
    logger.error('Error reading filelist {}'.format(filepath, err))


def my_collate(batch):
  batch = list(filter(lambda x: x is not None, batch))
  return default_collate(batch)

def extract_video_features(filepath):
  min_ts = 64
  max_ts = 64
  input_features_dim = 2048
  f = np.load(filepath)
  # sample a window from all timesteps
  # of atleast self.min_ts and atmost self.max_ts
  if f.shape[0] < min_ts:
    return None

  ts_begin = random.randint(0, f.shape[0] - min_ts)
  ts_end = random.randint(ts_begin + min_ts,
                          min(ts_begin + max_ts, f.shape[0]))

  # get the features from that window
  temp_win = np.zeros([max_ts, input_features_dim])
  temp_win_rev = np.zeros_like(temp_win)
  mask_rev = np.zeros([max_ts])
  mask_rev[0: ts_end - ts_begin] = 1
  temp_win[0: ts_end - ts_begin, :] = f[ts_begin:ts_end, :]
  temp_win_rev[0: ts_end - ts_begin, :] = f[ts_begin:ts_end, :][::-1, :]

  return filepath, temp_win, temp_win_rev, mask_rev