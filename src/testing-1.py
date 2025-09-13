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


import speech_recognition as sr
import io
import pyaudio


def preprocess_sample_tweets(sample_tweets, word_to_index, max_tweet_length=50):
    processed_tweets = []
    for tweet in sample_tweets:
        words = tweet.lower().split()
        tweet_indices = [word_to_index.get(word, 0) for word in words]
        processed_tweets.append(tweet_indices)
    processed_tweets = pad_sequences(processed_tweets, maxlen=max_tweet_length, padding='post')
    return processed_tweets

def predict_with_model(model, x_test, y_test=None):
    prediction_probability = model.predict(x_test)
    
    # Convert probabilities to predicted labels
    for i, _ in enumerate(prediction_probability):
        confidence = np.max(prediction_probability[i])
        accuracy = confidence * 100
        predicted = np.argmax(prediction_probability[i])  # Get the class with the highest probability
        

    # If ground truth is provided, print evaluation statistics
    if y_test is not None:
        utils.print_statistics(y_test, y_pred)

    return predicted, round(accuracy,2)


def test_sample_tweets(model, sample_tweets, word_to_index, max_tweet_length=50):
    sample_tweets_cleaned = data_proc.clean(sample_tweets)
    
    processed_tweets = preprocess_sample_tweets(sample_tweets_cleaned, word_to_index, max_tweet_length)
    result , accuracy = predict_with_model(model, processed_tweets)
    sarcasm = "Sarcastic" if result == 1 else "Non-Sarcastic"
    print("Tweet:", sample_tweets)
    print(f"Prediction:{{'{sarcasm}', {accuracy:.2f}}}")
    
    
    
def audio_to_text():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Capture audio from the microphone
    with sr.Microphone() as source:
        print("Please speak something...")
        # Adjust for ambient noise and record audio
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    # Convert audio to text
    try:
        text = recognizer.recognize_google(audio)
        print(f"Detected text: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

if __name__ == "__main__":
    path = os.getcwd()[:os.getcwd().rfind('\\')]
    word_to_index = np.load('word_to_index.npy', allow_pickle=True).item()

    # Load model architecture and weights
    with open('D:/sarcasm/models/dnn_models/model_json/standard_model.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("D:/sarcasm/models/dnn_models/best/standard_model.weights.h5")

    # Get user choice for input method
    user_choice = input("Enter 'audio' to record audio or 'text' for text input: ").strip().lower()

    if user_choice == 'audio':
        detected_text = audio_to_text()
        if detected_text:
            test_sample_tweets(model, [detected_text], word_to_index)

    elif user_choice == 'text':
        sample_tweets = []
        while True:
            tweet = input("Enter a tweet (or type 'exit' to finish): ")
            if tweet.lower() == 'exit':
                break
            sample_tweets.append(tweet)

        test_sample_tweets(model, sample_tweets, word_to_index)
    else:
        print("Invalid choice. Please enter 'audio' or 'text'.")
