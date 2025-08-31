# Efficient training and uncertainty estimation in DNNs with RHO-LOSS and ABNN

## Project Overview

[cite_start]This repository contains the code and resources for the course project carried as part of **CS772: Probabilistic Machine Learning** at IIT Kanpur under Prof. Piyush Rai[cite: 1, 2, 3, 4, 5, 6]. [cite_start]The project focuses on tackling two significant challenges in deep learning: the inefficiency of training on massive, noisy datasets and the lack of predictive uncertainty in standard Deep Neural Networks (DNNs)[cite: 15, 16, 19]. [cite_start]The experiments were conducted on the **CIFAR-10** and **CIFAR-100** datasets, with CIFAR-100 being a more complex challenge due to its higher number of classes[cite: 98].

Our pipeline integrates two primary techniques:

1.  [cite_start]**RHO-LOSS**: A prioritized data selection strategy that improves training efficiency by focusing on data points that are learnable, worth learning, and not yet learned[cite: 10, 26, 27]. [cite_start]This method filters out redundant and noisy samples to achieve higher accuracy faster[cite: 18].

2.  [cite_start]**Adaptable Bayesian Neural Networks (ABNN)**: A framework used to convert a pre-trained DNN into a Bayesian Neural Network (BNN)[cite: 12]. [cite_start]By transforming normalization layers into their Bayesian counterparts and fine-tuning the model, we can equip the network with the ability to quantify predictive uncertainty with minimal computational overhead[cite: 21, 46, 47].

[cite_start]The goal of this project was to create a cohesive framework that efficiently trains a robust DNN and subsequently equips it with reliable uncertainty estimates[cite: 23].

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

[cite_start]By prioritizing more informative data points, RHO-LOSS not only sped up the training process but also led to better final model accuracy compared to the standard uniform data selection method[cite: 103, 115, 126]. [cite_start]The model trained with RHO-LOSS consistently achieved lower loss and higher accuracy in fewer epochs[cite: 103].

**Test Accuracy Comparison**

| Dataset | Uniform Selection | RHO-LOSS Selection |
| :--- | :---: | :---: |
| CIFAR-10 | [cite_start]85.67% [cite: 116] | [cite_start]**89.23%** [cite: 116] |
| CIFAR-100 | [cite_start]56% [cite: 127] | [cite_start]**57.78%** [cite: 127] |

### ABNN: Calibrated Uncertainty with Minimal Overhead

[cite_start]Converting the RHO-LOSS-trained DNNs to ABNNs significantly improved the models' predictive uncertainty with only a minor trade-off in accuracy[cite: 131]. [cite_start]The improvement is quantified by the **Negative Log-Likelihood (NLL)** metric, where a lower value indicates better-calibrated uncertainty estimates[cite: 130].

**Performance Comparison (DNN vs. ABNN)**

| Dataset | Model | Test Accuracy | NLL (Test Set) |
| :--- | :--- | :---: | :---: |
| **CIFAR-10** | DNN | [cite_start]**89.23%** [cite: 132] | [cite_start]1.78 [cite: 132] |
| | A-BNN | [cite_start]87.08% [cite: 132] | [cite_start]**0.59** [cite: 132] |
| **CIFAR-100**| DNN | [cite_start]**57.78%** [cite: 135] | [cite_start]4.07 [cite: 135] |
| | A-BNN | [cite_start]57.76% [cite: 135] | [cite_start]**1.93** [cite: 135] |

---

## Authors

This project was developed by:
* [cite_start]Aastha Punjabi (210017) [cite: 2]
* [cite_start]Rohan Batra (210868) [cite: 3]
* [cite_start]Priya (210771) [cite: 5]
* [cite_start]Sharvani (210960) [cite: 6]

