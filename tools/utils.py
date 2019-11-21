import logging
import sys

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
