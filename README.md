# Chest X-rays image classification

## Overview

This machine learning project, mainly developed using TensorFlow, aims to predict if a person has heart conditions, based on features as lifestyle and general health conditions.

## Table of Contents

- [EDA and preprocessing](#EDA_and_preprocessing)
- [Model Training](#model-training)
- [Results](#Results)


The repository is organized as follows:

## EDA and preprocessing

### 0_Datasets

- A CSV file containing more than 300000 records, with 27373 heart disease cases and 292422 records of people with no heart condition. The evident class imbalance makes it necessary to undersample the 'no-heart-disease' class.

### 1_EDA

- **Exploratory_data_analysis:** In this notebook we analyzed, using the whole dataset, the correlation between heart conditions and the different patologies or lifestyle that influence them.

- **Undersampled_Exploratory_data_analysis:** Here, we conducted the same analysis as in the previous notebook, utilizing only the undersampled dataset.

### 2_preprocessing

- **preprocessing_features:** A notebook used just to create a second dataset with some features removed.

## Model Training

### 3_models

- **Baseline:** Creating a baseline using KNN, Logistic regression and Naive Bayes.

- **Dense_model:** A dense model obtained with parameter optimization on the dropout, number of neurons, number of layers and different optimizers.

- **CNN_model:** A convolutional neural network obtained with parameter optimization on the dropout, Batch normalization, kernel_size,number of filters, number of layers and different optimizers.

- **preprocessing_functions:** A python script with all the functions necessary to preprocess (train test split, undersampling, encoding and normalization) the dataset. 

## Results

### 5_results

-**results_evaluation:** In this notebook we evaluate the results of the best performing model.

