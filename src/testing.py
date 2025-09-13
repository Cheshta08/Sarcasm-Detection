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


from tensorflow.keras.utils import register_keras_serializable
from keras import initializers, activations
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from keras.utils import plot_model
from numpy.random import seed

def preprocess_sample_tweets(sample_tweets, word_to_index, max_tweet_length=110):
    processed_tweets = []
    for tweet in sample_tweets:
        words = tweet.lower().split()
        tweet_indices = [word_to_index.get(word, 0) for word in words]
        processed_tweets.append(tweet_indices)
    processed_tweets = pad_sequences(processed_tweets, maxlen=max_tweet_length, padding='post')
    return processed_tweets

def test_sample_tweets(model, sample_tweets, word_to_index, max_tweet_length=110):
    # Preprocess the sample tweets
    sample_tweets_cleaned=data_proc.clean(sample_tweets)
    print(sample_tweets)
    processed_tweets = preprocess_sample_tweets(sample_tweets_cleaned, word_to_index, max_tweet_length)
    
    predictions = model.predict(processed_tweets)
    
    for tweet, prediction in zip(sample_tweets, predictions):
        # Determine the predicted class based on the threshold (assuming binary classification)
        pred_class = 'Sarcastic' if prediction[0] > 0.5 else 'Not Sarcastic'
        print(f"Tweet: {tweet}")
        print(f"Prediction: {pred_class, prediction[0]} \n")


if __name__ == "__main__":
    path = os.getcwd()[:os.getcwd().rfind('\\')]
    to_write_filename = path + '/stats/test_tweets_analysis.txt'
    utils.initialize_writer(to_write_filename)
    from keras.models import model_from_json
    

# # Load model architecture from JSON
#     with open('D:/sarcasm/models/dnn_models/model_json/attention_model.json', 'r') as json_file:
#       model_json = json_file.read()

# # Recreate the model with custom layer
#     model = model_from_json(model_json, custom_objects={'MyAttentionLayer': MyAttentionLayer})

# # Load model weights
#     model.load_weights("D:/sarcasm/models/dnn_models/best/attention_model.keras")

    with open('D:/sarcasm/models/dnn_models/model_json/standard_model.json', 'r') as json_file:
      model_json = json_file.read()

# Recreate the model with custom layer
    model = model_from_json(model_json)

# Load model weights
    model.load_weights("D:/sarcasm/models/dnn_models/best/standard_model.keras")

    sample_tweets = [
    "Oh, you’re right. I totally forgot how much I enjoy getting up at 5 AM every day 😅 #NotAMorningPerson",
    "Today workout was intense, but I am feeling really accomplished. 🏋️",
    "Awesome, now my coffee is cold. Exactly what I needed this morning ☕️",
    "Fantastic, the elevator is out of order again. Guess I’ll get some exercise today 🚶‍♂️",
    "Excited for the upcoming concert this weekend! 🎸🎉",
    "Great, the one day I decide to bring an umbrella, it’s sunny all day! 🌞",
    "Had a productive day at work and managed to complete all my tasks. 🎉"   
    ]
    word_to_index = np.load('word_to_index.npy', allow_pickle=True).item()
    
    
    test_sample_tweets(model, sample_tweets,word_to_index)

    