import os
import numpy as np
import data_processing as data_proc  # Assuming this is your custom module for data cleaning
from keras.models import load_model, model_from_json
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

# Define the preprocessing function for tweets
def preprocess_sample_tweets(sample_tweets, word_to_index, max_tweet_length=110):
    processed_tweets = []
    for tweet in sample_tweets:
        words = tweet.lower().split()
        tweet_indices = [word_to_index.get(word, 0) for word in words]
        processed_tweets.append(tweet_indices)
    processed_tweets = pad_sequences(processed_tweets, maxlen=max_tweet_length, padding='post')
    return processed_tweets

# Define the function for testing and predicting sarcasm in sample tweets
def test_sample_tweets(model, sample_tweets, word_to_index, max_tweet_length=110):
    # Preprocess the sample tweets
    sample_tweets_cleaned = data_proc.clean(sample_tweets)  # Clean the tweets using the clean function
    print("Cleaned Tweets: ", sample_tweets_cleaned)
    processed_tweets = preprocess_sample_tweets(sample_tweets_cleaned, word_to_index, max_tweet_length)
    
    predictions = model.predict(processed_tweets)
    
    for tweet, prediction in zip(sample_tweets, predictions):
        # Determine the predicted class based on the threshold (assuming binary classification)
        pred_class = 'Sarcastic' if prediction[0] > 0.5 else 'Not Sarcastic'
        print(f"Tweet: {tweet}")
        print(f"Prediction: {pred_class, prediction[0]} \n")

# Function to load the model
def load_lstm_model(model_json_path, model_weights_path):
    with open(model_json_path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)  # Load the model architecture
    model.load_weights(model_weights_path)  # Load the model weights
    return model

# Main script to load the model, process the data, and save predictions
if __name__ == "__main__":
    # Define the paths for the cleaned file and the word-to-index mapping
    cleaned_file_path = 'cleaned.xlsx'  # Path to the cleaned file
    word_to_index = np.load('word_to_index.npy', allow_pickle=True).item()  # Load the word-to-index mapping
    
    # Load the trained LSTM model
    model_json_path = 'D:/sarcasm/models/dnn_models/model_json/lstm_model.json'
    model_weights_path = 'D:/sarcasm/models/dnn_models/best/lstm_model.keras'
    model = load_lstm_model(model_json_path, model_weights_path)

    # Load the cleaned data from Excel
    df_cleaned = pd.read_excel(cleaned_file_path)

    # Extract the 'text' column from the dataframe
    sample_tweets = df_cleaned['text'].tolist()

    # Predict sarcasm for each tweet in the cleaned dataset
    predictions = model.predict(preprocess_sample_tweets(data_proc.clean(sample_tweets), word_to_index))

    # Append the predictions to the dataframe
    df_cleaned['is_Sarcastic'] = ['Sarcastic' if pred[0] > 0.5 else 'Not Sarcastic' for pred in predictions]

    # Save the dataframe with predictions back to Excel
    output_file_path = 'cleaned_with_predictions.xlsx'
    df_cleaned.to_excel(output_file_path, index=False)
    print(f"Predictions saved to {output_file_path}")
