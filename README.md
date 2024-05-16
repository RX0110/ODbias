# Project Title: Outlier Detection Bias Analysis

## Description
This project explores the bias inherent in various outlier detection algorithms including Local Outlier Factor (LOF), Isolation Forest, Deep Autoencoder (DeepAE), and Fairod. The study focuses on understanding the sources of algorithmic bias through data-centric factors in simulated environments.

## Algorithms Used
- Local Outlier Factor (LOF): Identifies outliers by measuring the local deviation of density of a given sample with respect to its neighbors.
- Isolation Forest: Uses an ensemble of random trees to isolate and detect anomalies in data.
- Deep Autoencoder (DeepAE): Employs a neural network structure that learns to compress and decompress data, identifying outliers by reconstruction error.
- Fairod: A fairness-enhanced outlier detection model that incorporates fairness regularization to mitigate bias.

## Installation
Clone this repository and install the required packages.

## Usage
This section provides instructions on how to run analyses using different methods included in the project. Before running each script, make sure to switch the method or data type as needed at the beginning of each file.

### LOF / IF
- **To run the test:** Execute the following command in your terminal:
  ```bash
  python [filename].py
Replace `[filename]` with the actual script name. Modify the method (LOF/IF) and the data type (cluster/scatter) in the script settings.

### DeepAE
- **To run the test:** Execute the following command in your terminal:
  ```bash
  python deepae_whole.py
Adjust the bias type and data type at the start of the file as required.

### FairOD
- **To run the test:** Execute the following command in your terminal:
  ```bash
  python fair.py
Adjust the bias type and data type at the start of the file as required.

### Results Storage
Results from the executions are stored in the results folder structured as follows:
- **result subfolder** : performance metrics that used to plot
- **new_data**: data including prediction of DeepAE and FairOD
- **hparam_value**: hyperparameter of DeepAE and FairOD
  
## Data
The project uses simulated data to inject known biases and evaluate the impact on the detection performance of each algorithm. The data is structured to highlight sample size disparity, under-representation, measurement noise and obfuscation across different groups.

## Results
The results folder contains the output from each algorithm's run, detailing performance metrics such as AUROC score, along with fairness metrics.
