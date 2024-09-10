import os, time, itertools
from sklearn import preprocessing
import extract_baseline_features
import extract_ml_features as extract_features
import utils, classifiers
import data_processing as data_proc
import numpy as np
from sklearn.ensemble import VotingClassifier
from scipy import sparse
from sklearn.decomposition import PCA

# Settings for the upcoming ML model
pragmatic = True
lexical = True
pos_grams = True
sentiment = True
topic = True
similarity = True
pos_ngram_list = [1]
ngram_list = [1]
embedding_dim = 100
word2vec_map = utils.load_vectors(filename='glove.6B.%dd.txt' % embedding_dim)

# Set the values for the portion of data
n_train = 3000
n_test = 500

def baseline(tweets_train, train_labels, tweets_test, test_labels):
    # Import the subjectivity lexicon
    subj_dict = data_proc.get_subj_lexicon()

    types_of_features = ['1', '2', '3', 'ngrams']
    for t in types_of_features:
        start = time.time()
        utils.print_model_title("Classification using feature type " + t)
        
        if t == '1':
            x_train_features = extract_baseline_features.get_features1(tweets_train, subj_dict)
            x_test_features = extract_baseline_features.get_features1(tweets_test, subj_dict)
        
        elif t == '2':
            x_train_features = extract_baseline_features.get_features2(tweets_train, subj_dict)
            x_test_features = extract_baseline_features.get_features2(tweets_test, subj_dict)
        
        elif t == '3':
            x_train_features = extract_baseline_features.get_features3(tweets_train, subj_dict)
            x_test_features = extract_baseline_features.get_features3(tweets_test, subj_dict)
        
        elif t == 'ngrams':
            ngram_map, x_train_features = extract_baseline_features.get_ngram_features(tweets_train, n=1)
            x_test_features = extract_baseline_features.get_ngram_features_from_map(tweets_test, ngram_map, n=1)

        # Convert to sparse matrices
        x_train_features_sparse = sparse.csr_matrix(x_train_features)
        x_test_features_sparse = sparse.csr_matrix(x_test_features)

        # Get the class ratio
        class_ratio = utils.get_classes_ratio_as_dict(train_labels)

        # Train and evaluate Linear SVM and Logistic Regression
        svm_model = classifiers.linear_svm(x_train_features_sparse, train_labels, x_test_features_sparse, test_labels, class_ratio, max_iter=5000)
        lr_model = classifiers.logistic_regression(x_train_features_sparse, train_labels, x_test_features_sparse, test_labels, class_ratio, max_iter=5000)
        
        # Ensemble model
        ensemble_model = VotingClassifier(estimators=[('svm', svm_model), ('lr', lr_model)], voting='hard')
        ensemble_model.fit(x_train_features_sparse, train_labels)
        predictions = ensemble_model.predict(x_test_features_sparse)
        
        end = time.time()
        print("Completion time of the baseline model with features type %s: %.3f s = %.3f min" % (t, (end - start), (end - start) / 60.0))

def reduce_dimensions(x_train, x_test, n_components=100):
    pca = PCA(n_components=n_components)
    
    # Fit PCA on the training data and apply the same transformation to the test data
    x_train_reduced = pca.fit_transform(x_train.toarray())
    x_test_reduced = pca.transform(x_test.toarray())
    
    return x_train_reduced, x_test_reduced

def ml_model(train_tokens, train_pos, y_train, test_tokens, test_pos, y_test):

    print("Processing TRAIN SET features...\n")
    start = time.time()
    train_pragmatic, train_lexical, train_pos, train_sent, train_topic, train_sim = extract_features.get_feature_set(
        train_tokens, train_pos, pragmatic=pragmatic, lexical=lexical,
        ngram_list=ngram_list, pos_grams=pos_grams, pos_ngram_list=pos_ngram_list,
        sentiment=sentiment, topic=topic, similarity=similarity, word2vec_map=word2vec_map)
    end = time.time()
    print("Completion time of extracting train models: %.3f s = %.3f min" % ((end - start), (end - start) / 60.0))

    print("Processing TEST SET features...\n")
    start = time.time()
    test_pragmatic, test_lexical, test_pos, test_sent, test_topic, test_sim = extract_features.get_feature_set(
        test_tokens, test_pos, pragmatic=pragmatic, lexical=lexical,
        ngram_list=ngram_list, pos_grams=pos_grams, pos_ngram_list=pos_ngram_list,
        sentiment=sentiment, topic=topic, similarity=similarity, word2vec_map=word2vec_map)
    end = time.time()
    print("Completion time of extracting test models: %.3f s = %.3f min" % ((end - start), (end - start) / 60.0))

    # Get all features together
    all_train_features = [train_pragmatic, train_lexical, train_pos, train_sent, train_topic, train_sim]
    all_test_features = [test_pragmatic, test_lexical, test_pos, test_sent, test_topic, test_sim]

    # Choose your feature options: you can run on all possible combinations of features
    sets_of_features = 6
    feature_options = list(itertools.product([False, True], repeat=sets_of_features))
    feature_options = feature_options[1:]  # skip over the option in which all entries are false

    for option in feature_options:
        train_features = [{} for _ in range(len(train_tokens))]
        test_features = [{} for _ in range(len(test_tokens))]
        utils.print_features(option, ['Pragmatic', 'Lexical-grams', 'POS-grams', 'Sentiment', 'LDA topics', 'Similarity'])

        # Make a feature selection based on the current feature_option choice
        for i, o in enumerate(option):
            if o:
                for j, example in enumerate(all_train_features[i]):
                    train_features[j] = utils.merge_dicts(train_features[j], example)
                for j, example in enumerate(all_test_features[i]):
                    test_features[j] = utils.merge_dicts(test_features[j], example)

        # Vectorize and scale the features using sparse matrices
        x_train_sparse, x_test_sparse = utils.extract_features_from_dict(train_features, test_features)
        
        # Reduce dimensions using PCA
        x_train_reduced, x_test_reduced = reduce_dimensions(x_train_sparse, x_test_sparse)

        print("Shape of the x train set (%d, %d)" % (len(x_train_reduced), len(x_train_reduced[0])))
        print("Shape of the x test set (%d, %d)" % (len(x_test_reduced), len(x_test_reduced[0])))

        # Run and ensemble SVM and Logistic Regression models
        svm_model = classifiers.linear_svm(x_train_reduced, y_train, x_test_reduced, y_test, max_iter=5000)
        lr_model = classifiers.logistic_regression(x_train_reduced, y_train, x_test_reduced, y_test, max_iter=5000)

        ensemble_model = VotingClassifier(estimators=[('svm', svm_model), ('lr', lr_model)], voting='hard')
        ensemble_model.fit(x_train_reduced, y_train)
        predictions = ensemble_model.predict(x_test_reduced)

        end = time.time()
        print("Completion time of the Ensemble model: %.3f s = %.3f min" % ((end - start), (end - start) / 60.0))

    return ensemble_model

def test_model(ensemble_model, sample_tweets):
    # Preprocess the sample tweets
    sample_tokens, sample_pos = data_proc.preprocess_sample_tweets(sample_tweets)
    sample_features = extract_features.get_feature_set(sample_tokens, sample_pos, 
                                                       pragmatic=pragmatic, lexical=lexical, 
                                                       ngram_list=ngram_list, pos_grams=pos_grams, 
                                                       pos_ngram_list=pos_ngram_list, sentiment=sentiment, 
                                                       topic=topic, similarity=similarity, 
                                                       word2vec_map=word2vec_map)

    # Vectorize and scale the features using sparse matrices
    sample_features_sparse = utils.extract_features_from_dict(sample_features)
    sample_features_reduced = reduce_dimensions(sample_features_sparse)

    # Predict using the ensemble model
    predictions = ensemble_model.predict(sample_features_reduced)

    # Display the results
    for tweet, pred in zip(sample_tweets, predictions):
        label = "Sarcastic" if pred == 1 else "Non-Sarcastic"
        print(f"Tweet: {tweet} => Prediction: {label}")

if __name__ == "__main__":
    path = os.getcwd()[:os.getcwd().rfind('\\')]
    to_write_filename = path + '/stats/ml_analysis-1.txt'
    utils.initialize_writer(to_write_filename)

    dataset = "ghosh"  # can be "ghosh", "riloff", "sarcasmdetection", "ptacek"
    train_tokens, train_pos, train_labels, test_tokens, test_pos, test_labels = data_proc.get_dataset(dataset)

    run_baseline = False

    if run_baseline:
        baseline(train_tokens, train_labels, test_tokens, test_labels)
    else:
        ensemble_model = ml_model(train_tokens, train_pos, train_labels, test_tokens, test_pos, test_labels)

        # Define some sample tweets for testing
        sample_tweets = [
            "Oh great, another Monday. I just love waking up early.",
            "Best vacation ever! Got stuck in traffic for 5 hours.",
            "This is the best food I've ever had in my life.",
            "Can't wait to get stuck in traffic again!"
        ]
        
        # Test the model with sample tweets
        test_model(ensemble_model, sample_tweets)
