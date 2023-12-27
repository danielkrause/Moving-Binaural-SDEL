
# Binaural Sound Source Distance Estimation and Localization for a Moving Listener

# add citation

## DATASETS

We study the method on [BinMov23: Binaural Dataset for Source Position Estimation with Head Rotation and Moving Listeners](https://doi.org/10.5281/zenodo.7689063).


## Getting Started

This repository consists of multiple Python scripts described below. 
* The `batch_feature_extraction.py` is a standalone wrapper script, that extracts the features, labels, and normalizes the training and test split features for a given dataset. Make sure you update the location of the downloaded datasets before.
* The `parameters.py` script consists of all the training, model, and feature parameters. If a user has to change some parameters, they have to create a sub-task with unique id here. Check code for examples.
* The `cls_feature_class.py` script has routines for labels creation, features extraction and normalization.
* The `cls_data_generator.py` script provides feature + label data in generator mode for training.
* The `models.py` script implements the different model architectures related to the paper mentioned above.
* The `cls_metric.py` script implements the metrics for localization and distance estimation.
* The `train_doa_sde_joint.py` is a wrapper script that trains SDEL models with a single task approach.
* The `train_doa_sde_joint_multi.py` is a wrapper script that trains SDEL models with a multi task approach.

### Prerequisites

The provided codebase has been tested on Python 3.9 and Torch 1.13


## License
This repository is an adapted version of [Vue]([https://github.com/vuejs/vue](https://github.com/sharathadavanne/doa-net)).
The repository is licensed under the [TAU License](LICENSE.md).
