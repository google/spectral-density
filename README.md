# Large Scale Spectral Density Estimation for Deep Neural Networks

This repository contains the distributed TensorFlow implementation of stochastic Lanczos
Quadrature for deep neural networks as used and described in [our paper](https://arxiv.org/abs/1901.10159).

## Training a CIFAR-10 model
1. Grab the [CIFAR-10 binary dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
2. Run training via `python experiments/cifar_train.py --train_data_path=<glob_of_data>`.
3. Run eval via `python experiments/cifar_eval.py --eval_data_path=<glob_of_data>`.


## Running distributed Lanczos algorithm
After the training jobs has written its checkpoints, we run `cifar_train.py` again, this time in `--train_mode=lanczos`. This also has the side effect of turning off some options that make the forward pass non-deterministic (`--shuffle_each_epoch=false` and `--augment=false`).

To run Lanczos on a checkpoint generated during training, we use `--checkpoint_to_load=...`. Our code should work as is in a distributed multi-GPU worker setting. If there are multiple GPU workers, then setting `--partition_data_per_worker` along with specifying `--task=<worker_id>` will speed the computation dramatically. Unfortunately, we do not currently have a multi-tower implementation, but we would love to work with you on one.

This process writes out two crucial numpy files: `tridiag_1` and `lanczos_vec_1` in the new `--train_log_dir=<path>` (Be sure to change this for the Lanczos phase if you don't want your checkpoints overwritten!). Note that we have very specific naming conventions for this flag to make it compatible with density calculator Jupyter notebook.


### Lanczos Parameters
Around 80-90 Lanczos_steps are certainly enough for most
models; we tend to set `--lanczos_draws=1` to enable parallelism over many
instances of the Lanczos job.

## Generating spectral densities
We run the Jupyter notebook to compute the spectral densities using the `tridiag_1` and `lanczos_vec_1` files computed earlier (we actually use this notebook to compute the density along an entire trajectory of checkpoints).


This is not an official Google product.
