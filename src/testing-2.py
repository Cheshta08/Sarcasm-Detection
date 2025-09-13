import os
import numpy as np
import data_processing as data_proc
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import speech_recognition as sr
import io
from pydub import AudioSegment

AudioSegment.converter = r"C:\Users\HP\Downloads\ffmpeg-2024-10-02-git-358fdf3083-essentials_build.zip\ffmpeg-2024-10-02-git-358fdf3083-essentials_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\Users\HP\Downloads\ffmpeg-2024-10-02-git-358fdf3083-essentials_build\ffmpeg-2024-10-02-git-358fdf3083-essentials_build\bin\ffprobe.exe"
def preprocess_sample_tweets(sample_tweets, word_to_index, max_tweet_length=110):
    processed_tweets = []
    for tweet in sample_tweets:
        words = tweet.lower().split()
        tweet_indices = [word_to_index.get(word, 0) for word in words]
        processed_tweets.append(tweet_indices)
    processed_tweets = pad_sequences(processed_tweets, maxlen=max_tweet_length, padding='post')
    return processed_tweets

def test_sample_tweets(model, sample_tweets, word_to_index, max_tweet_length=110):
    sample_tweets_cleaned = data_proc.clean(sample_tweets)
    print(sample_tweets)
    processed_tweets = preprocess_sample_tweets(sample_tweets_cleaned, word_to_index, max_tweet_length)
    
    predictions = model.predict(processed_tweets)
    
    for tweet, prediction in zip(sample_tweets, predictions):
        pred_class = 'Sarcastic' if prediction[0] > 0.5 else 'Not Sarcastic'
        print(f"Tweet: {tweet}")
        print(f"Prediction: {pred_class, prediction[0]} \n")

def convert_mp3_to_wav(mp3_file, wav_file):
    # Load the MP3 file
    audio = AudioSegment.from_mp3(mp3_file)
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

if __name__ == "__main__":
    path = os.getcwd()[:os.getcwd().rfind('\\')]
    word_to_index = np.load('word_to_index.npy', allow_pickle=True).item()

    # Load model architecture and weights
    with open('D:/sarcasm/models/dnn_models/model_json/lstm_model.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("D:/sarcasm/models/dnn_models/best/lstm_model.keras")

    # User input to choose audio or text
    user_choice = input("Enter 'audio' to input audio or 'text' for text input: ").strip().lower()

    if user_choice == 'audio':
        audio_file_path = input("Enter the path of the audio file: ")
        wav_file_path = "converted_audio.wav"  # Temporary WAV file path
        
        # Convert MP3 to WAV if necessary
        if audio_file_path.endswith('.mp3'):
            convert_mp3_to_wav(audio_file_path, wav_file_path)
            audio_path = wav_file_path
        else:
            audio_path = audio_file_path

        # Recognize audio
        detected_text = audio_to_text(audio_path)
        if detected_text:
            print(f"Detected text from audio: {detected_text}")
            test_sample_tweets(model, [detected_text], word_to_index)
        
        # Clean up: Remove the temporary WAV file
        if os.path.exists(wav_file_path):
            os.remove(wav_file_path)

    elif user_choice == 'text':
        # Get multiple tweets from user input
        sample_tweets = []
        while True:
            tweet = input("Enter a tweet (or type 'exit' to finish): ")
            if tweet.lower() == 'exit':
                break
            sample_tweets.append(tweet)

        test_sample_tweets(model, sample_tweets, word_to_index)
    else:
        print("Invalid choice. Please enter 'audio' or 'text'.")
