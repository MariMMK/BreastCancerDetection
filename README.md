# Breast Cancer Prediction using Artificial Neural Network

This project involves the creation of an Artificial Neural Network (ANN) to predict breast cancer using the Breast Cancer dataset from sklearn. The project includes data preprocessing, feature selection, model tuning using Grid Search, model training, and building an interactive web application using Streamlit.

## Table of Contents

- [Introduction](#introduction)
- [Project Setup](#project-setup)
- [Dataset Acquisition and Preparation](#dataset-acquisition-and-preparation)
- [Feature Selection](#feature-selection)
- [Model Tuning](#model-tuning)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Streamlit App](#streamlit-app)
- [Visualizations](#visualizations)
- [Usage](#usage)
- [Demo](#demo)
- [Conclusion](#conclusion)
- [License](#license)

## Introduction

This project uses the Breast Cancer dataset to predict whether a tumor is malignant or benign. The main steps involved are data acquisition and preparation, feature selection, model tuning using Grid Search, training an Artificial Neural Network (ANN), and creating a Streamlit web application for user interaction and prediction visualization.

## Project Setup

1. Clone the repository:

2. Set up a virtual environment:
    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
      ```bash
      .\venv\Scripts\activate
      ```
    - On macOS and Linux:
      ```bash
      source venv/bin/activate
      ```

4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Acquisition and Preparation

The Breast Cancer dataset from sklearn is used in this project. The dataset is loaded, split into training and testing sets, and preprocessed by applying feature selection and scaling.

## Feature Selection

Feature selection is performed using `SelectKBest` from sklearn's feature selection module to select the top 10 features based on the ANOVA F-value.

## Model Tuning

Grid Search Cross-Validation is used to tune the hyperparameters of the ANN model (`MLPClassifier` from sklearn.neural_network).

## Model Training and Evaluation

The best model from Grid Search is trained on the training data and evaluated on the test data. Performance metrics such as confusion matrix, classification report, and ROC curve are used to assess the model.

## Streamlit App

A Streamlit web application is created to allow users to input feature values and predict whether a tumor is malignant or benign. The app also displays the input data, prediction probability, and relevant visualizations.


## Usage

1. Run the Jupyter Notebook to preprocess the data, tune the model, train the model, and save the trained model:
    ```bash
    jupyter notebook project.ipynb
    ```

2. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Demo


![Screenshot 1](screenshots/screenshot1.png)
![Screenshot 2](screenshots/screenshot2.png)

## Conclusion

This project demonstrates the use of an Artificial Neural Network for predicting breast cancer, with a streamlined process for data preprocessing, feature selection, model tuning, training, and deployment using Streamlit.

