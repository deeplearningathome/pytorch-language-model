# Simple Word-based Language Model in PyTorch
This model is directly analagous to this [Tesnsorflow's LM](https://www.tensorflow.org/tutorials/recurrent).
In fact, the reader is directly taken from its older version

See this [blogpost.](http://deeplearningathome.com/2017/06/PyTorch-vs-Tensorflow-lstm-language-model.html)

## How to RUN:
```
python ptb-lm.py --data=[PATH_TO_DATA]
```
Default params should result in Test perplexity of ~78.04.
Your actual result will vary due to random initialization.
This basically matches results from TF's tutorial, only faster.

On GTX 1080 I am getting around 7,400 wps.

## Files
* lm.py - language model description
* reader.py - slightly older version of TF's PTB reader which yields numpy arrays as batches
* ptb-lm.py - driver script

## Requirements
* Python 3 (I used Anaconda distribution)
* PyTorch (I used 0.1.12)