
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os, time, utils, sys
import numpy as np
from tensorflow.keras.optimizers import Adam

import data_processing as data_proc
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, GRU, Bidirectional, Input, Multiply
from tensorflow.keras.layers import Layer
from keras.layers import Activation, Permute, RepeatVector, Lambda
from keras.utils import to_categorical
import tensorflow as tf
from keras import backend as K
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

from keras.utils import plot_model
from numpy.random import seed # Ensure your data_processing module is available.


# Function to preprocess tweets
def preprocess_sample_tweets(sample_tweets, word_to_index, max_tweet_length=50):
    processed_tweets = []
    for tweet in sample_tweets:
        words = tweet.lower().split()
        tweet_indices = [word_to_index.get(word, 0) for word in words]
        processed_tweets.append(tweet_indices)
    processed_tweets = pad_sequences(processed_tweets, maxlen=max_tweet_length, padding='post')
    return processed_tweets


# Advanced prediction function
def predict_with_model(model, x_test, y_test=None):
    y_pred = []
    prediction_probability = model.predict(x_test)
    print("Predicted probability length: ", len(prediction_probability))
    
    # Convert probabilities to predicted labels
    for i, _ in enumerate(prediction_probability):
        predicted = np.argmax(prediction_probability[i])  # Get the class with the highest probability
        y_pred.append(predicted)

    # If ground truth is provided, print evaluation statistics
    if y_test is not None:
        utils.print_statistics(y_test, y_pred)

    return y_pred


# Function to evaluate model on Excel file
def evaluate_model_on_excel(model, excel_file_path, word_to_index, max_tweet_length=50):
    # Load Excel data
    data = pd.read_excel(excel_file_path)
    tweets = data['Tweet']  # Column name in Excel file for tweets
    actual_labels = data['Actual_Label']  # Column name in Excel file for actual labels (0 or 1)

    # Preprocess tweets
    tweets_cleaned = data_proc.clean(tweets)  # Assuming 'utils.clean()' is your tweet-cleaning function
    processed_tweets = preprocess_sample_tweets(tweets_cleaned, word_to_index, max_tweet_length)

    # Predict with the model
    predicted_labels = predict_with_model(model, processed_tweets,actual_labels)

    # Append predictions to the dataframe
    data['Predicted_Label_lstm'] = predicted_labels

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
    with open('D:/sarcasm/models/dnn_models/model_json/lstm_model.json', 'r') as json_file:
        model_json = json_file.read()

    # Recreate the model with custom layer
    model = model_from_json(model_json)

    # Load model weights
    model.load_weights("D:/sarcasm/models/dnn_models/best/lstm_model.weights.h5")

    # Load word-to-index dictionary
    word_to_index = np.load('word_to_index.npy', allow_pickle=True).item()
    

    # Path to the Excel file
    excel_file_path = "subtitles.xlsx"  # Replace with your file path

    # Evaluate model on Excel data
    evaluate_model_on_excel(model, excel_file_path, word_to_index)