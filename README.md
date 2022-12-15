# Improving Generalization in Federated Learning by Seeking Flat Minima

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-generalization-in-federated/federated-learning-on-landmarks-user-160k)](https://paperswithcode.com/sota/federated-learning-on-landmarks-user-160k?p=improving-generalization-in-federated)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-generalization-in-federated/federated-learning-on-cifar-100-alpha-0-5)](https://paperswithcode.com/sota/federated-learning-on-cifar-100-alpha-0-5?p=improving-generalization-in-federated)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-generalization-in-federated/federated-learning-on-cifar-100-alpha-0-5-5)](https://paperswithcode.com/sota/federated-learning-on-cifar-100-alpha-0-5-5?p=improving-generalization-in-federated)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-generalization-in-federated/federated-learning-on-cityscapes)](https://paperswithcode.com/sota/federated-learning-on-cityscapes?p=improving-generalization-in-federated)

This repository contains the official implementation of
> Caldarola, D., Caputo, B., & Ciccone, M. [_Improving Generalization in Federated Learning by Seeking Flat Minima_](https://arxiv.org/abs/2203.11834), _European Conference on Computer Vision_ (ECCV) 2022.

## Abstract
Models trained in federated settings often suffer from degraded performances and fail at generalizing, especially
when facing heterogeneous scenarios. In this work, we investigate such behavior through the lens of geometry of the loss
and Hessian eigenspectrum, linking the model's lack of generalization capacity to the sharpness of the solution.
Motivated by prior studies connecting the sharpness of the loss surface and the generalization gap, we show that _i)_
training clients locally with Sharpness-Aware Minimization (SAM) or its adaptive version (ASAM) and _ii)_
averaging stochastic weights (SWA) on the server-side can substantially improve generalization in Federated Learning
and help bridging the gap with centralized models.
By seeking parameters in neighborhoods having uniform low loss, the model converges towards flatter minima and its
generalization significantly improves in both homogeneous and heterogeneous scenarios. Empirical results demonstrate the
effectiveness of those optimizers across a variety of benchmark vision datasets (e.g. CIFAR10/100, Landmarks-User-160k,
IDDA) and tasks (large scale classification, semantic segmentation, domain generalization).

<p align="center">
 <img src="loss_landscapes/loss_landscapes_fig1.png">
</p>


## Setup
### Environment
- Install conda environment (preferred): ```conda env create -f environment.yml```
- Install with pip (alternative): ```pip3 install -r requirements.txt```

### Weights and Biases
The code runs with WANDB. For setting up your profile, we refer you to the [quickstart documentation](https://docs.wandb.ai/quickstart). Insert your WANDB API KEY [here](https://github.com/debcaldarola/fedsam/blob/master/models/main.py#L24). WANDB MODE is set to "online" by default, switch to "offline" if no internet connection is available.

### Resources
All experiments run on one NVIDIA GTX-1070. If needed, you can specify the GPU ID [here](https://github.com/debcaldarola/fedsam/blob/master/models/main.py#L8).

### Data
Execute the following code for setting up the datasets:
```bash
conda activate torch10
cd data
chmod +x setup_datasets.sh
./setup_datasets.sh
```

## Datasets

1. CIFAR-100
  * **Overview**: Image Dataset based on [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) and [Federated Vision Datasets](https://github.com/google-research/google-research/tree/master/federated_vision_datasets)
  * **Details**: 100 users with 500 images each. Different combinations are possible, following Dirichlet's distribution
  * **Task**: Image Classification over 100 classes

2. CIFAR-10
  * **Overview**: Image Dataset based on [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and [Federated Vision Datasets](https://github.com/google-research/google-research/tree/master/federated_vision_datasets)
  * **Details**: 100 users with 500 images each. Different combinations are possible, following Dirichlet's distribution
  * **Task**: Image Classification over 10 classes


## Running experiments
Examples of commands for running the paper experiments (FedAvg/FedSAM/FedASAM w/ and w/o SWA) can be found in ```fedsam/paper_experiments```.
E.g. for CIFAR10 use the following command:
```bash
cd paper_experiments
chmod +x cifar10.sh
./cifar10.sh
```

## Bibtex
```text
@inproceedings{caldarola2022improving,
  title={Improving generalization in federated learning by seeking flat minima},
  author={Caldarola, Debora and Caputo, Barbara and Ciccone, Marco},
  booktitle={European Conference on Computer Vision},
  pages={654--672},
  year={2022},
  organization={Springer}
}
```
