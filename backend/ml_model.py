# This file will contain the logic for the machine learning component.
# It will analyze player statistics and pitch conditions to recommend the best players.
# The implementation will involve data preprocessing, model training, and prediction.

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function for data preprocessing
def preprocess_data(player_stats, pitch_conditions):
    # Ensure player_stats is a DataFrame
    if isinstance(player_stats, dict):
        player_stats = pd.DataFrame([player_stats])
    elif isinstance(player_stats, list):
        player_stats = pd.DataFrame(player_stats)
    elif not isinstance(player_stats, pd.DataFrame):
        raise ValueError("Invalid player_stats format. Expected a dictionary, list of dictionaries, or DataFrame.")

    # Ensure pitch_conditions is a DataFrame with a single row
    if isinstance(pitch_conditions, dict):
        pitch_conditions = pd.DataFrame([pitch_conditions])
    elif not isinstance(pitch_conditions, pd.DataFrame):
        raise ValueError("Invalid pitch_conditions format. Expected a dictionary or DataFrame.")

    # Replicate pitch_conditions for each player
    pitch_conditions_repeated = pd.concat([pitch_conditions] * len(player_stats), ignore_index=True)

    # Combine player stats and pitch conditions
    data = pd.concat([player_stats.reset_index(drop=True), pitch_conditions_repeated], axis=1)

    # Handle missing values
    data.fillna(0, inplace=True)

    # Convert all columns to numeric, if possible
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='ignore')

    # Remove non-numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data = data[numeric_columns]

    # Ensure data is not empty
    if data.empty:
        raise ValueError("Preprocessed data is empty. Check input data.")

    return data

# Function for training the model
def train_model(X):
    model = RandomForestClassifier()
    # Use a dummy target variable for unsupervised learning
    y_dummy = np.zeros(X.shape[0])
    model.fit(X, y_dummy)
    return model

# Function for making predictions
def predict_best_players(model, X_test):
    logging.info(f"X_test shape: {X_test.shape}")
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_test)
        if probabilities.shape[1] == 2:
            predictions = probabilities[:, 1]  # Return probabilities of positive class for binary classification
        else:
            predictions = np.mean(probabilities, axis=1)  # Average probabilities for multi-class
    else:
        # For models without predict_proba, use predict and normalize
        predictions = model.predict(X_test)
        if predictions.ndim > 1:
            predictions = np.mean(predictions, axis=1)

    # Ensure predictions are in the range [0, 1]
    predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min() + 1e-8)

    logging.info(f"Predictions shape: {predictions.shape}")
    return predictions.flatten()  # Ensure 1D array output

# The main function is removed as it's not needed for the web application.
# The model training and prediction are now handled in the Flask routes.

