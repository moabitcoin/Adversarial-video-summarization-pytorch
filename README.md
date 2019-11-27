# Introduction
We want to qualitatively estimate the visual diversity within our drive data. For measuring scene diversity we want to use visual semantic similarity of the drives. F.example drives in high traffic density vs drives of vehicle waiting at traffic lights. 
# Training/Validation Data
We use pre-trained ResNet50 features extracted from BDD100K videos as our per-frame visual representation. To reduce computation we downsample the [BDD100K](https://bdd-data.berkeley.edu) videos(30fps@1280x720) as 5fps@640x360. This generates `TMAX`~200 vectors of `D=2048` dimensions per video. We use temporal windows of length `T=64` randomly sampled from `TMAX` positions.

## Video representations
![](imgs/lstm-auto-encoder.png)
We use a LSTM Autoencoder to model video representation generator. The core idea uses this [paper](http://www.cs.toronto.edu/~rsalakhu/papers/video_lstm.pdf). An encoder LSTM reads in input visual features of shape `[T, D]` and generate a summary vector (or [thought vector](https://gabgoh.github.io/ThoughtVectors)) of shape `S=128`. The decoder LSTM reads in the thought vector and reproduces the input visual features. We regress the reproduced visual features against the input visual features with MSE. The core idea being that the visual features which are redundant between frames are compressed with the Autoencoder. `T, S, D` are hyper-parameters we control to affect model complexity/performance/runtime. The Autoencoder trained at this stage forms the `eLSTM` and `dLSTM` for the next stage.


## Video summarisation
![](imgs/adversarial-lstm.png)

The core idea use this [paper](https://mahasseb.github.io/files/2017/cvpr_video_summarization.pdf) `TODO`

## Getting Started
This is an example of how you may give instructions on setting up your project locally. To get a local copy up and running follow these simple example steps.

### Train encoder LSTM (bidirectional = False)
```
python auto_encoder/train_encoder.py —train_features_list <train_features_list_path> --log_dir <save_logs_dir_path> —model_save_dir <path_to_model_dir>
```
### Train decoder LSTM (bidirectional = True)
```
python auto_encoder/train_decoder.py —train_features_list <train_features_list_path> --log_dir <save_logs_dir_path> —model_save_dir <path_to_model_dir>
```
`learning_rate = 1e-4` <br>
`batch_size = 256` <br>
`num_workers = 12` <br>
`n_epochs = 300` <br>
`save_interval = 1000 step` <br>

If you want to change this values you can add the variables as command line arguments.

### Build Features index
```
python scripts/build_index.py —model_path <path_to_model> —features_list <path to text file containing resnet features files list> —index_path <path_where_to_save_index>
```
### Query by Video Features
```
python scripts/matcher.py --index_path <path_to_index> --model_path <path_to_model> -features_query <path_to_video_features_file>
```
<code>ex: python scripts/matcher.py -i drives-index.pck -m encoder_lstm.pth-43000 -q 000000_016839bf-0247-432f-8af6-5d33a12a0341-video.npy </code>

## License

Copyright © 2019 MoabitCoin

Distributed under the MIT License (MIT).