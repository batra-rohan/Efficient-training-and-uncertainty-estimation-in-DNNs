# Efficient DNN Training and Uncertainty Estimation with RHO-LOSS and ABNN

## Project Overview

This repository contains the code and resources for the course project carried as part of **CS772: Probabilistic Machine Learning** at IIT Kanpur under Prof. Piyush Rai. The project focuses on tackling two significant challenges in deep learning: the inefficiency of training on massive, noisy datasets and the lack of predictive uncertainty in standard Deep Neural Networks (DNNs). The experiments were conducted on the **CIFAR-10** and **CIFAR-100** datasets, with CIFAR-100 being a more complex challenge due to its higher number of classes.

Our pipeline integrates two primary techniques:

1.  **RHO-LOSS**: A prioritized data selection strategy that improves training efficiency by focusing on data points that are learnable, worth learning, and not yet learned. This method filters out redundant and noisy samples to achieve higher accuracy faster.

2.  **Adaptable Bayesian Neural Networks (ABNN)**: A framework used to convert a pre-trained DNN into a Bayesian Neural Network (BNN). By transforming normalization layers into their Bayesian counterparts and fine-tuning the model, we can equip the network with the ability to quantify predictive uncertainty with minimal computational overhead.

The goal of this project was to create a cohesive framework that efficiently trains a robust DNN and subsequently equips it with reliable uncertainty estimates.

---

## How to Reproduce the Results

The project pipeline involves two main stages. First, train the baseline DNNs using the RHO-LOSS strategy. Second, convert these trained DNNs into ABNNs for uncertainty analysis.

### Step 1: Train the Initial DNN Models

This phase involves training ResNet-18 models on the CIFAR-10 and CIFAR-100 datasets using the RHO-LOSS data selection strategy to improve training efficiency and accuracy.

* **For CIFAR-10**: Run the `rho_loss_cifar10.ipynb` notebook. This script trains the irreducible loss model, then trains the target ResNet-18 model on CIFAR-10, and saves the final model checkpoint.
* **For CIFAR-100**: Run the `rho_loss_cifar100.ipynb` notebook. This performs the same training pipeline but is configured for the CIFAR-100 dataset.

These initial training steps produce the DNN models that will be used for the next phase.

### Step 2: Convert DNNs to Adaptable Bayesian Neural Networks (ABNNs)

This phase involves converting the previously trained DNNs into ABNNs to equip them with uncertainty estimation capabilities.

* **For CIFAR-10 ABNN**: Run the `CIFAR10_DNN_TO_ABNN.ipynb` notebook. This script loads the trained CIFAR-10 model checkpoint, replaces its normalization layers with Bayesian Normalization Layers (BNLs), and fine-tunes the new ABNN model for a few epochs.
* **For CIFAR-100 ABNN**: Run the `CIFAR100_DNN_TO_ABNN.ipynb` notebook. This follows the same procedure, converting the trained CIFAR-100 model into an ABNN and then fine-tuning it.

---

## File Descriptions

This repository contains four main Jupyter notebooks that cover the entire experimental pipeline for both the CIFAR-10 and CIFAR-100 datasets.

### 1. `rho_loss_cifar10.ipynb`
* **Purpose**: Implements the complete training pipeline for a **ResNet-18 model on the CIFAR-10 dataset** using the RHO-LOSS methodology.
* **Process**:
    1.  Trains an "irreducible loss model" on a holdout set to estimate the inherent noise level of each data point.
    2.  Trains the primary target model using a data selection strategy based on the calculated irreducible losses.
    3.  Saves the final trained model checkpoint and the calculated irreducible losses for later use.

### 2. `rho_loss_cifar100.ipynb`
* **Purpose**: Follows the same pipeline as the one for CIFAR-10 but is configured for the more complex **CIFAR-100 dataset**.
* **Process**: It trains an irreducible loss model and then a target ResNet-18 model on CIFAR-100, demonstrating the effectiveness of the RHO-LOSS approach on a larger dataset.

### 3. `CIFAR10_DNN_TO_ABNN.ipynb`
* **Purpose**: Takes the trained ResNet-18 model from `rho_loss_cifar10.ipynb` and converts it into an **Adaptable Bayesian Neural Network (ABNN)**.
* **Process**:
    1.  Loads the pre-trained DNN checkpoint.
    2.  Replaces standard `BatchNorm2d` layers in the ResNet-18 architecture with custom **Bayesian Normalization Layers (BNL)**.
    3.  Fine-tunes the new BNN for a few epochs to learn the uncertainty parameters.
    4.  Evaluates the ABNN's performance, focusing on accuracy and Negative Log-Likelihood (NLL).

### 4. `CIFAR100_DNN_TO_ABNN.ipynb`
* **Purpose**: Performs the ABNN conversion for the **CIFAR-100 model** trained in `rho_loss_cifar100.ipynb`.
* **Process**: It follows the same conversion and fine-tuning steps as the CIFAR-10 version, adapting the final DNN to provide uncertainty estimates for the 100-class problem.

---

## Key Results

Our integrated pipeline demonstrates significant improvements in both training efficiency and model reliability.

### RHO-LOSS: Faster Training and Higher Accuracy

By prioritizing more informative data points, RHO-LOSS not only sped up the training process but also led to better final model accuracy compared to the standard uniform data selection method. The model trained with RHO-LOSS consistently achieved lower loss and higher accuracy in fewer epochs.

**Test Accuracy Comparison**

| Dataset | Uniform Selection | RHO-LOSS Selection |
| :--- | :---: | :---: |
| CIFAR-10 | 85.67% | **89.23%** |
| CIFAR-100 | 56% | **57.78%** |

### ABNN: Calibrated Uncertainty with Minimal Overhead

Converting the RHO-LOSS-trained DNNs to ABNNs significantly improved the models' predictive uncertainty with only a minor trade-off in accuracy. The improvement is quantified by the **Negative Log-Likelihood (NLL)** metric, where a lower value indicates better-calibrated uncertainty estimates.

**Performance Comparison (DNN vs. ABNN)**

| Dataset | Model | Test Accuracy | NLL (Test Set) |
| :--- | :--- | :---: | :---: |
| **CIFAR-10** | DNN | **89.23%** | 1.78 |
| | A-BNN | 87.08% | **0.59** |
| **CIFAR-100**| DNN | **57.78%** | 4.07 |
| | A-BNN | 57.76% | **1.93** |

---

## Authors

This project was developed by:
* Aastha Punjabi (210017)
* Rohan Batra (210868)
* Priya (210771)
* Sharvani (210960)

---

## References

The methodologies used in this project are based on the following research papers:

* **RHO-LOSS:** Mindermann, S. et al. (2022). [Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt](https://proceedings.mlr.press/v162/mindermann22a/mindermann22a.pdf). *Proceedings of the International Conference on Machine Learning (ICML)*.

* **ABNN:** Franchi, G. et al. (2024). [Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from Pre-trained Models](https://openaccess.thecvf.com/content/CVPR2024/papers/Franchi_Make_Me_a_BNN_A_Simple_Strategy_for_Estimating_Bayesian_CVPR_2024_paper.pdf). *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
