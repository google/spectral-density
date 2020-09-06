# Large Scale Spectral Density Estimation for Deep Neural Networks

This repository contains two implementations of the stochastic Lanczos Quadrature algorithm for deep neural networks as used and described in [Ghorbani, Krishnan and Xiao, _An Investigation into Neural Net Optimization via Hessian Eigenvalue Density_ (ICML 2019)](https://arxiv.org/abs/1901.10159).

To run the example notebooks, please first `pip install tensorflow_datasets`.

## TensorFlow Implementation
The main class that runs distributed Lanczos algorithm is [`LanczosExperiment`](https://github.com/google/spectral-density/blob/f0d3f1446bb1c200d9200cbdc67407e3f148ccba/tf/lanczos_experiment.py#L33). The Jupyter [notebook](https://github.com/google/spectral-density/blob/master/tf/mnist_spectral_density.ipynb) demonstrates how to use this class. 

In addition to single machine (potentially multiple-GPU setups), this implementation is also suitable for multi-GPU multi-worker setups. The crucial step is manually partitioning the input data across the available GPUs.

The algorithm outputs two numpy files: `tridiag_1` and `lanczos_vec_1` which are the tridiagonal matrix and Lanczos vectors. The tridiagonal matrix can then be used to generate spectral densities using [`tridiag_to_density`](https://github.com/google/spectral-density/blob/f0d3f1446bb1c200d9200cbdc67407e3f148ccba/jax/density.py#L120).

## Jax Implementation (by [Justin Gilmer](https://github.com/jmgilmer))
The Jax version is fantastic for fast experimentation (especially in conjunction with [trax](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/trax)). The Jupyter [notebook](https://github.com/google/spectral-density/blob/f0d3f1446bb1c200d9200cbdc67407e3f148ccba/jax/mnist_hessian_example.ipynb) demonstrates how to run Lanczos in Jax.

The main function is [`lanczos_alg`](https://github.com/google/spectral-density/blob/f0d3f1446bb1c200d9200cbdc67407e3f148ccba/jax/lanczos.py#L27), which returns a tridiagonal matrix and Lanczos vectors. The tridiagonal matrix can then be used to generate spectral densities using [`tridiag_to_density`](https://github.com/google/spectral-density/blob/f0d3f1446bb1c200d9200cbdc67407e3f148ccba/jax/density.py#L120).

## Differences between implementations
1. The TensorFlow version performs Hessian-vector product accumulation and the actual Lanczos algorithm in float64, whereas the Jax version performs all calculation in float32.
2. The TensorFlow version targets multi-worker distributed setups, whereas the Jax version targets single worker (potentially multi-GPU) setups.

This is not an official Google product.
