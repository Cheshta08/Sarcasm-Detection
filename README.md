# Sarcasm Detection in Tweets

## Overview
This project focuses on detecting sarcasm in tweets by using a combination of traditional machine learning models and advanced deep learning architectures. The solution involves several key steps, such as data preprocessing, feature extraction, and model training. The dataset includes multilingual data, primarily in English and Hindi (in Roman script), to address sarcasm detection in diverse linguistic contexts.

## Key Components

### 1. Data Preprocessing
- **Tokenization**: Splitting tweets into individual tokens (words) to structure the data.
- **POS Tagging**: Assigning part-of-speech (POS) tags to tokens for syntactic analysis.
- **Label Encoding**: Converting sarcasm labels into numerical format to facilitate model training.
- **Language Detection and Translation**: Detecting Hindi tweets written in Roman script and translating them into English for consistency.
- **Emoji Analysis**: Translating emojis into descriptive text to capture emotional context.

### 2. Feature Extraction
- **Pragmatic Features**: Extracting features such as tweet length, capitalized words, user mentions, etc.
- **Sentiment Features**: Deriving sentiment scores using tools like Vader and emoji sentiment lexicons.
- **Syntactic Features**: Extracting features based on POS tags, such as counts of nouns, verbs, and adjectives.
- **Topic Modeling**: Applying Latent Dirichlet Allocation (LDA) to uncover underlying topics in the tweets.

### 3. Model Training and Evaluation
A variety of machine learning and deep learning models were trained using the extracted features:

#### Machine Learning Models
- **Support Vector Machines (SVM)**: Used for binary classification of sarcastic vs. non-sarcastic tweets.
- **Logistic Regression**: Used to predict sarcasm by analyzing extracted features.

#### Deep Learning Models
- **Deep Neural Networks (DNN)**: A fully connected deep learning model for learning complex tweet patterns.
- **Long Short-Term Memory Networks (LSTM)**: A recurrent neural network model for capturing sequential tweet context.
- **Attention Models**: Used to focus on important sections of tweets for better sarcasm detection.

### 4. Model Evaluation
The models were evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-Score

## Technologies Used
- **Programming Language**: Python
- **Libraries/Frameworks**:
  - NLP: `NLTK`, `SpaCy`, `Vader`
  - Machine Learning: `Scikit-learn`
  - Deep Learning: `TensorFlow`, `Keras`
  - Topic Modeling: `Gensim` (LDA)
  - Data Handling: `Pandas`, `NumPy`

## Models Used
- Support Vector Machines (SVM)
- Logistic Regression
- Deep Neural Networks (DNN)
- Long Short-Term Memory Networks (LSTM)
- Attention Models

## Future Work
- Experiment with more advanced deep learning models to further improve sarcasm detection accuracy.
- Fine-tune feature extraction and scaling methods for better performance across diverse datasets.
- The model focused primarily on mixed Hindi-English language, but can be extended to work on across other languages.
- Model only categorises the input as 'Sarcastic' or 'Non-Sarcastic', so can be further extended to classify specific types of sarcasm

  ## Contribution
- Edit by Priyanshi Modi
