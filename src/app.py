from flask import Flask, request, jsonify, render_template
import numpy as np
import json
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
import speech_recognition as sr
import io
import pyaudio
from pydub import AudioSegment

from keras.utils import plot_model
from numpy.random import seed # Ensure your data_processing module is available.


app = Flask(__name__)

def convert_to_wav(file, wav_file):
    # Load the MP3 file
    audio = AudioSegment.from_file(file)
    # Export as WAV file
    audio.export(wav_file, format="wav")

def audio_to_text(audio_path):
    # Initialize recognizer
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)  # Read the entire audio file
        try:
            # Convert audio to text
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

# Load the model
def load_model():
    with open('D:/sarcasm/models/dnn_models/model_json/cnn + lstm_model.json', 'r') as json_file:
        model_json = json_file.read()

    # Recreate the model with custom layer
    model = model_from_json(model_json)

    # Load model weights
    model.load_weights("D:/sarcasm/models/dnn_models/best/cnn + lstm_model.keras")

    # Load word-to-index dictionary
    global word_to_index
    word_to_index = np.load('word_to_index.npy', allow_pickle=True).item()

    return model

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
    print("Predicted probability length: ", len(prediction_probability))
    
    # Convert probabilities to predicted labels
    for i, _ in enumerate(prediction_probability):
        print(prediction_probability[i])
        confidence = np.max(prediction_probability[i])
        print(confidence*100)
        accuracy = confidence * 100
        predicted = np.argmax(prediction_probability[i])  # Get the class with the highest probability
        

    # If ground truth is provided, print evaluation statistics
    if y_test is not None:
        utils.print_statistics(y_test, y_pred)

    return predicted, round(accuracy,2)


model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_text', methods=['POST'])
def predict_text():
    data = request.json
    sentence = data['sentence']
    cleaned = data_proc.clean([sentence])
    processed = preprocess_sample_tweets(cleaned, word_to_index)
    result , accuracy = predict_with_model(model, processed)
    sarcasm = "Sarcastic" if result == 1 else "Non-Sarcastic"
    return jsonify({'result': sarcasm , 'accuracy':accuracy})
    
    
@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    file = request.files['audio']
    print("filereached")
    audio_path = 'uploads/' + file.filename
    file.save(audio_path)

    wav_file_path = "audio.wav"  # Change this to your desired path
    
    # Convert MP3 to WAV if necessary
    if not audio_path.endswith('.wav'):
        convert_to_wav(audio_path, wav_file_path)
        audio_path = wav_file_path
    

    # Recognize audio
    detected_text = audio_to_text(audio_path)
    if detected_text:
        print(f"Detected text from audio: {detected_text}")
        cleaned = data_proc.clean([detected_text])
        processed = preprocess_sample_tweets(cleaned, word_to_index)
        result, accuracy = predict_with_model(model, processed)
        sarcasm = "sarcastic" if result == 1 else "non-sarcastic"
        return jsonify({'result': sarcasm , "accuracy": accuracy})
    else:
        return jsonify({'result': "error", 'message': "Audio detection failed"}), 400    
        
    
if __name__ == '__main__':
    app.run(debug=True)