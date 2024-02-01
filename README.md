
# eFedGauss: A Federated Approach to Fuzzy Multivariate Gaussian Clustering
**This paper is in the review process.**

## Overview
Evolving Federated Gaussian clustering (eFedGauss) addresses the challenge of determining the number of clusters in datasets that vary widely, a common issue in federated learning. Our method dynamically adjusts the number of clusters, making it versatile for different types of data, including complex and high-dimensional datasets like those in credit card fraud detection. Our tests, which include synthetic data, the Iris Flower dataset, and credit card fraud detection, demonstrate eFedGauss's superiority over traditional clustering methods in federated settings. This repository provides all necessary resources for implementing and exploring the capabilities of eFedGauss.

## Repository Structure

The eFedGauss project is organized as follows:

- `model/`: Contains the implementation of the eFedGauss algorithm and its associated operations.
  - `clustering_operations/`: Functions related to the clustering process.
  - `consequence_operations/`: Operations for handling the consequence output of the model.
  - `eFedGauss/`: The main eFedGauss algorithm implementation.
  - `federated_operations/`: Federated learning specific operations.
  - `math_operations/`: Mathematical functions used across the model (distance computation).
  - `merging_mechanism/`: Logic for merging clusters.
  - `model_operations/`: Core operations for managing the lifecycle of the clustering model.
  - `removal_mechanism/`: Methods for removing clusters.

- `utils/`: Utility scripts to support the main algorithm.
  - `utils_dataset/`: Utilities for data handling and preprocessing.
  - `utils_metrics/`: Metrics for evaluating the model's performance.
  - `utils_plots/`: Plotting functions to visualize the results.
  - `utils_train/`: Training and testing.

- `credit_card_fraud_experiment/`: Experimental setup and results for the credit card fraud detection.

- `iris_flower_experiment/`: Experimental setup and results for the Iris Flower classification.

- `synthetic_experiment/`: Experimental setup and results for testing with synthetic data.

## Getting Started

To get started with eFedGauss, follow the setup instructions below.

### Setting Up the Environment

**Create and activate a Conda environment using the following commands:**

\```bash
conda create -n eFedGauss python=3.10
conda activate eFedGauss
\```

### Installing PyTorch and Related Packages

**Install PyTorch and related packages with the following command:**

\```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
\```

### Installing Additional Requirements

**Install the remaining requirements from the `requirements.txt` file:**

\```bash
pip install -r requirements.txt
\```

## Usage

**Run the experiments using the Jupyter notebooks provided:**

- `credit_card_fraud_experiment.ipynb`: For credit card fraud detection.
- `iris_flower_experiment.ipynb`: For Iris Flower classification.
- `synthetic_experiment.ipynb`: For synthetic data experiments.

## Requirements

The required packages are listed in the `requirements.txt` file.

## Contribution

Contributions are welcome. Please follow the standard fork-and-pull request workflow.

## License

eFedGauss is under GNU General Public License v3 (GPLv3). See `LICENSE` for more details.
