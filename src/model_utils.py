import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def normalize_features(X_train, X_val):
    """
    Normalize features using StandardScaler for both the training and validation sets.

    Parameters:
        X_train (DataFrame or array-like): Training set features.
        X_val (DataFrame or array-like): Validation set features.

    Returns:
        X_train_normalized (DataFrame): Normalized training set features.
        X_val_normalized (DataFrame): Normalized validation set features.
        norm_model (StandardScaler): StandardScaler model used to normalize the data.
    """
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Find the columns for validation set
    val_cols = X_val.columns

    # Fit and transform on the training set, then transform the validation set
    X_train_normalized = scaler.fit_transform(X_train[val_cols])
    X_val_normalized = scaler.transform(X_val)
    
    # normalisation model
    norm_model = scaler
    # Convert the normalized NumPy arrays to DataFrames
    X_train_normalized = pd.DataFrame(X_train_normalized,index=X_train.index, columns=X_val.columns)
    X_val_normalized = pd.DataFrame(X_val_normalized, index=X_val.index, columns=X_val.columns)

    return X_train_normalized, X_val_normalized, norm_model

def save_model_and_normalization(model, feat, normalization, file_path):
    """
    Save a trained model and data normalization object into a single file.

    Parameters:
    - model: Trained machine learning model.
    - normalization: Data normalization object (e.g., StandardScaler).
    - file_path: File path to save the model and normalization.

    Returns:
    None
    # Example usage:
    # Assuming you have a trained model 'my_model' and a normalization object 'my_normalization'
    # Replace these with your actual model and normalization objects

    # save_model_and_normalization(my_model, my_normalization, 'model_and_normalization_file.joblib')
    """
    # Create a dictionary to store model and normalization information
    model_and_normalization = {
        'model': model,
        'feat': feat,
        'normalization': normalization,
    }

    # Save the dictionary to a file using joblib
    joblib.dump(model_and_normalization, file_path)

def load_model_and_normalization(file_path):
    """
    Load a trained model and data normalization object from a file.

    Parameters:
    - file_path: File path to load the model and normalization.

    Returns:
    - model: Trained machine learning model.
    - feat: features used for training the model
    - normalization: Data normalization object (e.g., StandardScaler).
    """
    # Load the dictionary from the file
    model_and_normalization = joblib.load(file_path)

    # Retrieve the model and normalization from the dictionary
    model = model_and_normalization['model']
    feat = model_and_normalization['feat']
    normalization = model_and_normalization['normalization']

    return model, feat, normalization

def normalization(norm_model, df):
    '''This function normalizes the features and returns dataframe'''
    return pd.DataFrame(norm_model.transform(df),index=df.index, columns=df.columns)

def predict_within_range(model, data, min_value=None, max_value=None):
    """
    Make predictions using a machine learning model while ignoring rows with NaN values
    and setting predictions outside a specified range to NaN.

    Parameters:
    - model: Trained machine learning model with a predict method.
    - data: Pandas DataFrame with NaN values.
    - min_value: Minimum valid prediction value. If None, no minimum check is performed.
    - max_value: Maximum valid prediction value. If None, no maximum check is performed.

    Returns:
    - np.ndarray: Predictions.
    """
    # Copy the original DataFrame to avoid modifying it
    data_no_nan = data.copy()

    # Identify rows with NaN values
    nan_rows = data_no_nan.isnull().any(axis=1)

    # Remove rows with NaN values
    features = data_no_nan[~nan_rows]

    # Make predictions using the filtered features
    predictions = np.squeeze(model.predict(features))     # squeeze to convert 2D array to 1D array for PLSR

    # Set predictions outside the specified range to NaN
    if min_value is not None or max_value is not None:
        predictions[(min_value is not None) & (predictions < min_value)] = np.nan
        predictions[(max_value is not None) & (predictions > max_value)] = np.nan

    # Create a new DataFrame with NaN values in the target column
    nan_predictions = pd.DataFrame(np.nan, index=data.index, columns=['predicted_CLS']) 

    # Fill in the predictions for rows without NaN values
    nan_predictions.loc[~nan_rows, 'predicted_CLS'] = predictions

    return nan_predictions['predicted_CLS']