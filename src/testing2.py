
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os, time, utils, sys
import numpy as np
import data_processing as data_proc
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, LSTM, GRU, Bidirectional, Input, Multiply
from tensorflow.keras.layers import Layer
from keras.layers import Activation, Permute, RepeatVector, Lambda
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
from dl_models import MyAttentionLayer 
from tensorflow.keras.models import model_from_json
import pandas as pd
from tensorflow.keras.utils import register_keras_serializable
from keras import initializers, activations
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle

from keras.utils import plot_model
from numpy.random import seed # Ensure your data_processing module is available.


# Function to preprocess tw


# Function to evaluate model on Excel file
def evaluate_model_on_excel(model, excel_file_path):
    # Load Excel data
    data = pd.read_excel(excel_file_path)
    tweets = data['Tweets']  # Column name in Excel file for tweets
    actual_labels = data['Actual_Label']  # Column name in Excel file for actual labels (0 or 1)

    
    # Predict with the model
    predictions = model.predict(tweets)
    
    for tweet, prediction in zip(sample_tweets, predictions):
        pred_class = 'Sarcastic' if prediction > 0.5 else 'Not Sarcastic'
        print(f"Tweet: {tweet}")
        print(f"Prediction: {pred_class, prediction} \n")


    # Append predictions to the dataframe
    data['Predicted_Label'] = predictions

    # Save the updated dataframe back to the Excel file
    with pd.ExcelWriter(excel_file_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        data.to_excel(writer, index=False)

    # Calculate metrics
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels)
    recall = recall_score(actual_labels, predicted_labels)
    f1 = f1_score(actual_labels, predicted_labels)
    conf_matrix = confusion_matrix(actual_labels, predicted_labels)

    # Print results
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(actual_labels, predicted_labels))


if __name__ == "__main__":
    # Initialize log file
    path = os.getcwd()[:os.getcwd().rfind('\\')]
    to_write_filename = path + '/stats/analysis.txt'
    utils.initialize_writer(to_write_filename)
    
    # Load model architecture from JSON
    model_path = 'svm_model.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Path to the Excel file
    excel_file_path = "shuffled_file.xlsx"  # Replace with your file path

    # Evaluate model on Excel data
    evaluate_model_on_excel(model, excel_file_path)
    