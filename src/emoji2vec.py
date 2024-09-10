from __future__ import print_function
import numpy as np
from pandas import read_csv, concat, DataFrame
import os, utils
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tensorflow.keras.preprocessing.text import Tokenizer


from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Input, LSTM, Embedding, Dense, GRU, Dropout, Reshape, Bidirectional
from keras.callbacks import ModelCheckpoint
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
symbola_font_path = r'C:\Users\HP\Downloads\symbola\Symbola.ttf'

# Add the Symbola font to the font manager
font_manager.fontManager.addfont(symbola_font_path)

# Set Symbola as the default font for the plot
plt.rcParams['font.family'] = 'Symbola'


# Define the path to the resources and make some settings
path = os.getcwd()[:os.getcwd().rfind('\\')]
emoji_positive = path + '/res/emoji/emoji_positive_samples.txt'
emoji_negative = path + '/res/emoji/emoji_negative_samples.txt'
emoji_freq = path + '/res/emoji/emoji_frequencies.txt'
maximum_length = 15
embedding_dim = 100         # valid: 50, 100, 200, 300
glove_filename = 'glove.6B.%dd.txt' % embedding_dim
emoji2vec_visualization = path + '/models/emoji2vec/emoji_emb_viz_%dd.csv' % embedding_dim
emoji2vec_weights = path + '/models/emoji2vec/weights_%dd.keras' % embedding_dim
emoji2vec_embeddings = path + '/models/emoji2vec/emoji_embeddings_%dd.txt' % embedding_dim


# Get a list of emojis ordered by their frequency of appearing in tweets
def get_emoji_frequencies():
    lines = utils.load_file(emoji_freq)
    frequencies = [line.split()[0] for line in lines]
    return frequencies


# Visualize the TSNE representation of the emoji embeddings
def visualize_emoji_embeddings(top=800):
    # Load a list of most popular emojis and plot those
    popular_emojis = get_emoji_frequencies()[:top]

    # Load the data frame
    df = read_csv(emoji2vec_visualization)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Get the data you want ot plot
    x_values = []
    y_values = []
    for index, row in df.iterrows():
        if row['emoji'] in popular_emojis:
            x_values.append(row['x'])
            y_values.append(row['y'])
            ax.text(row['x'], row['y'], row['emoji'], fontname='Symbola')
    plt.scatter(x_values, y_values, marker='o', alpha=0.0)
    plt.title('t-SNE Visualization of Emoji Embeddings')
    plt.grid()
    plt.show()
    plt.savefig(path + "/plots/emoji2vec/emoji_%dd_vis.png" % embedding_dim)


# Define emoji2vec DNN model
from keras.layers import Input, Embedding, Reshape, Bidirectional, LSTM, Dense, Concatenate, Dropout
from keras.models import Model

def emoji2vec_model(embedding_matrix, emoji_vocab_size, word_vocab_size):
    # Define emoji model
    emoji_input = Input(shape=(1,), name='emoji_input')
    emoji_embedding = Embedding(emoji_vocab_size + 1, embedding_dim, trainable=True)(emoji_input)
    emoji_output = Reshape((embedding_dim,))(emoji_embedding)
    emoji_model = Model(inputs=emoji_input, outputs=emoji_output, name='emoji_model')

    # Define word model
    word_input = Input(shape=(maximum_length,), name='word_input')
    word_embedding = Embedding(word_vocab_size + 1, embedding_dim, weights=[embedding_matrix], trainable=False)(word_input)
    word_lstm = Bidirectional(LSTM(embedding_dim, dropout=0.5, return_sequences=False))(word_embedding)
    word_model = Model(inputs=word_input, outputs=word_lstm, name='word_model')

    # Combine outputs
    combined = Concatenate()([emoji_output, word_lstm])

    # Add dense layers
    dense_output = Dense(embedding_dim * 2, activation='relu')(combined)
    dropout = Dropout(0.5)(dense_output)
    final_output = Dense(2, activation='softmax')(dropout)

    # Define the final model
    model = Model(inputs=[emoji_input, word_input], outputs=final_output, name='combined_model')

    return emoji_model, word_model, model



# Solely based on emoji descriptions, obtain the emoji2vec representations for all possible emojis
def get_emoji2vec():
    # Load the emoji data - both true and false descriptions
    pos_emojis = read_csv(emoji_positive, sep='\t', engine='python', encoding='utf_8', names=['description', 'emoji'])
    neg_emojis = read_csv(emoji_negative, sep='\t', engine='python', encoding='utf_8', names=['description', 'emoji'])

    print('Number of true emoji descriptions: %d' % len(pos_emojis))
    print('Number of false emoji descriptions: %d' % len(neg_emojis))

    # Set the labels to 1 (for true descriptions) and 0 (for false descriptions)
    pos_emojis['label'] = 1
    neg_emojis['label'] = 0

    # Concatenate and shuffle negative and positive examples of emojis
    all_emojis = concat([pos_emojis, neg_emojis]).sample(frac=1, random_state=144803)

    # Group all emojis in positive examples by descriptions
    emoji_grouping = pos_emojis.groupby('emoji')['description'].apply(lambda x: ', '.join(x))
    grouped_by_description = DataFrame({'emoji': emoji_grouping.index, 'description': emoji_grouping.values})

    # Build an emoji vocabulary and map each emoji to an index (beginning from 1)
    emojis = grouped_by_description['emoji'].values
    emoji_to_index = {emoji: index + 1 for emoji, index in zip(emojis, range(len(emojis)))}
    index_to_emoji = {index: emoji for emoji, index in emoji_to_index.items()}
    emoji_vocab_size = len(emoji_to_index)
    print('Total number of unique emojis: %d' % emoji_vocab_size)

    # Build a word vocabulary and map each emoji to an index (beginning from 1)
    descriptions = all_emojis['description'].values
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(descriptions.tolist())
    word_sequences = tokenizer.texts_to_sequences(descriptions.tolist())
    word_to_index = tokenizer.word_index
    index_to_word = {index: word for word, index in word_to_index.items()}
    word_vocab_size = len(word_to_index)
    print('Total number of unique words found in emoji descriptions: %d' % word_vocab_size)

    # Load GLoVe word embeddings
    print("Loading GLoVe...")
    word2vec_map = utils.load_vectors(glove_filename)

    # Prepare the word-embedding matrix
    embedding_matrix = utils.get_embedding_matrix(word2vec_map, word_to_index, embedding_dim, init_unk=False)
    print('Number of non-existent word-embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    # Prepare training data
    train_emoji = np.array([emoji_to_index[emoji] for emoji in all_emojis['emoji'].values])
    train_words = pad_sequences(word_sequences, maxlen=maximum_length)
    labels = np.array([[0, 1] if label == 0 else [1, 0] for label in all_emojis['label'].values])

    print('Shape of emoji data:', train_emoji.shape)
    print('Shape of emoji description data:', train_words.shape)
    print('Shape of label tensor:', labels.shape)
    print('Number of emojis:', emoji_vocab_size)

    # Build the emoji DNN model
    print("Building the emoji2vec model...")
    emoji_model, word_model, model = emoji2vec_model(embedding_matrix, emoji_vocab_size, word_vocab_size)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    print(model.summary())

    # Train a model if one hasn't been trained yet
    if not os.path.exists(emoji2vec_weights):
        print("Training the emoji2vec model...")
        callbacks = [ModelCheckpoint(emoji2vec_weights, monitor='val_categorical_accuracy', save_best_only=True)]
        history = model.fit([train_emoji, train_words], labels, epochs=50,
                            validation_split=0.1, verbose=1, callbacks=callbacks)
        # Plot accuracy and loss
        utils.plot_training_statistics(history, path + "/plots/emoji2vec/emoji2vec_%dd" % embedding_dim,
                                       also_plot_validation=True, acc_mode='categorical_accuracy', loss_mode='loss')
    print("Loading the trained weights of the emoji2vec model...")
    model.load_weights(emoji2vec_weights)

    # Ensure emoji_model is correctly defined
    print("Inspecting emoji_model...")
    print(emoji_model.summary())
    print(emoji_model.layers)
    # Access weights from the embedding layer in emoji_model
    embedding_layer = emoji_model.get_layer('embedding')
    weights = embedding_layer.get_weights()[0]

    # Load the pre-trained weights and get the embeddings
    print("Loading the trained weights of the emoji2vec model...")
    model.load_weights(emoji2vec_weights)
    weights = emoji_model.layers[1].get_weights()[0]

    # Get the emoji2vec mapping
    emoji2vec = {}
    for e, w in zip(grouped_by_description['emoji'], weights[1:]):
        emoji2vec[e] = w

    # Get the emoji embeddings and save them to file
    if not os.path.exists(emoji2vec_embeddings):
        embeddings = DataFrame(weights[1:])
        embeddings = concat([grouped_by_description['emoji'], embeddings], axis=1)
        embeddings.to_csv(emoji2vec_embeddings, sep=' ', header=False, index=False)

    # Get the t-SNE representation
    if not os.path.exists(emoji2vec_visualization):
        tsne = TSNE(n_components=2, perplexity=30, init='pca', n_iter=5000)
        trans = tsne.fit_transform(weights)

        # Save the obtained emoji visualization
        visualization = DataFrame(trans[1:], columns=['x', 'y'])
        visualization['emoji'] = grouped_by_description['emoji'].values
        visualization.to_csv(emoji2vec_visualization)

        # Visualize the embeddings as a tsne figure
        visualization.plot('x', 'y', kind='scatter', grid=True)
        plt.savefig(path + '/plots/emoji2vec/tsne_%dd.pdf' % embedding_dim)

    return emoji2vec

if __name__ == "__main__":
    emoji2vec = get_emoji2vec()

    # Plot an emoji map
    visualize_emoji_embeddings()
    

    # Get some intuition whether the model is good by seeing what analogies it can make based on what it learnt
    utils.make_analogy("👑", "🚹", "🚺", emoji2vec)  # Crown - Man + Woman
    utils.make_analogy("👑", "👦", "👧", emoji2vec)  # Crown - Boy + Girl
    utils.make_analogy("💵", "🇺🇸", "🇬🇧", emoji2vec)
    utils.make_analogy("💵", "🇺🇸", "🇪🇺", emoji2vec)
    utils.make_analogy("👪", "👦", "👧", emoji2vec)
    utils.make_analogy("🕶", "☀️", "⛈", emoji2vec)  # Sunglasses - Sun + Cloud
    