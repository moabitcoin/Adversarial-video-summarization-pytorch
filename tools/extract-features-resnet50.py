import argparse
import os
import ssl

import numpy as np
import tensorflow as tf
from skvideo.io import vreader
from tensorflow.keras.applications import resnet50
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context
# 3 is warning only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def read_file_list(filepath):
  try:

    with open(filepath, 'r') as pfile:
      vidfiles = pfile.readlines()
      vidfiles = [v.strip() for v in vidfiles]

    return vidfiles

  except Exception as err:

    print('Error reading {}, {}'.format(filepath, err))


def build_model():
  input_tensor_shape = (None, 224, 224, 3)

  input_img = tf.compat.v1.placeholder(tf.float32,
                                       input_tensor_shape,
                                       name='input_img')

  x = resnet50.preprocess_input(input_img)

  resnet = resnet50.ResNet50(weights='imagenet', include_top=False,
                             pooling='avg')

  output_tensor = resnet(x)

  return input_img, output_tensor


def generate_batches(video_path, batch_size=64, video_options=None):
  vid = vreader(str(video_path), outputdict={'-s': '224x224'})

  batch = []

  for frame in vid:

    if len(batch) == batch_size:
      batch = []

    batch.append(frame)

    if len(batch) == batch_size:
      yield np.array(batch)

  yield np.array(batch)


def run_feature_extractor(vid_list, destination, overwrite=False):
  with tf.compat.v1.Session() as sess:

    input_image, output_tensor = build_model()

    pbar = tqdm(vid_list, desc="Videos", unit="vid", ascii=True)

    for vid in pbar:
      features = []
      batches = generate_batches(vid)
      for idx, np_batch in enumerate(batches):
        output_val = sess.run([output_tensor], {input_image: np_batch})[0]
        features.append(output_val)

      features = np.concatenate(features, axis=0)

      # print(output_val.shape)
      # [N, 2048]
      filename = os.path.basename(vid)
      basename, _ = os.path.splitext(filename)
      feature_file = os.path.join(destination, basename + '.npy')
      print(feature_file)

      np.save(feature_file, features, allow_pickle=False)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('TF Feature extractor using keras API'
                                   'On ResNet50 model pre classification')

  parser.add_argument('-v', '--vid-file-list', dest='vid_file_list',
                      help='Text file of video list')
  parser.add_argument('-d', '--destination', dest='destination',
                      help='Location where to store the features')
  parser.add_argument('-o', '--overwrite', dest='overwrite', default=False,
                      help='Overwrite existing features')

  args = parser.parse_args()

  vid_list = read_file_list(args.vid_file_list)

  run_feature_extractor(vid_list, args.destination, args.overwrite)
