
# eFedGauss: A Federated Approach to Fuzzy Multivariate Gaussian Clustering
This paper is in the review process.

## Overview
eFedGauss introduces a novel method for data clustering and classification in federated learning. It addresses the challenge of determining the number of clusters in datasets that vary widely, a common issue in federated learning. Our method dynamically adjusts the number of clusters, making it versatile for different types of data, including complex and high-dimensional datasets like those in credit card fraud detection. Our tests, which include synthetic data, the Iris Flower dataset, and credit card fraud detection, demonstrate eFedGauss's superiority over traditional clustering methods in federated settings. This repository provides all necessary resources for implementing and exploring the capabilities of eFedGauss.

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
  - `utils_tables/`: Functions to generate result tables.
  - `utils_train/`: Training and testing.

- `credit_card_fraud_experiment/`: Experimental setup and results for the credit card fraud detection.

- `iris_flower_experiment/`: Experimental setup and results for the Iris Flower classification.

- `synthetic_experiment/`: Experimental setup and results for testing with synthetic data.

### Setting Up the Environment

Create and activate a Conda environment using the following commands:
conda create -n eFedGauss python=3.10
conda activate eFedGauss

### Installing PyTorch and Related Packages

Install PyTorch, torchvision, torchaudio, and the appropriate version of CUDA:

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

### Installing Additional Requirements

Next, install the other necessary packages listed in the `requirements.txt` file:

install requirements.txt

## Usage

The experiments are implemented in the files: 

- `credit_card_fraud_experiment.ipynb` for the credit card fraud detection experiment.
- `iris_flower_experiment.ipynb` for the iris flower classification experiment.
- `synthetic_experiment.ipynb` for the experiment with synthetic data.

## Requirements

The Python package requirements are listed in `requirements.txt`.

## Contribution

Contributions to this repository are welcome. Please fork the project, create a feature branch, commit your enhancements or fixes, and submit a pull request.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial (CC BY-NC) license. This allows for sharing and adapting the work non-commercially as long as appropriate credit is given and any new creations are non-commercially used and licensed under identical terms.

For more information, see [Creative Commons Attribution-NonCommercial (CC BY-NC) license](https://creativecommons.org/licenses/by-nc/4.0/).