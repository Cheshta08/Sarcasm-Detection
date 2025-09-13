import re, os, itertools, string,nltk

import contractions
from googletrans import Translator

# os.environ["GROQ_API_KEY"] = "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"
# client = Groq( )

from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import words
import numpy as np
import time
import emoji
import utils
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from vocab_helpers import implicit_emoticons, slang, \
    wikipedia_emoticons, emotiocons_to_emojis

path = os.getcwd()[:os.getcwd().rfind('\\')]
dict_filename = "word_list.txt"
word_filename = "word_list_freq.txt"

# def hinglish_to_english(hinglish_text):
#     # Split the text by emojis and special characters, capturing them as separate segments
#     segments = re.split(r'([^\w\s])', hinglish_text)  # Splits by non-word characters (e.g., emojis, punctuation)
#     processed_segments = []
    
#     for segment in segments:
#         if re.search(r'[a-zA-Z]', segment):  # Check if the segment contains alphabetic characters (indicating words)
#             try:
#                 # Construct the prompt for the model
#                 prompt = f"Translate '{segment}' from Hinglish to English: and don't give explanation only give me the translated sentence as output"
#                 response = client.chat.completions.create(
#                     model="llama3-70b-8192",
#                     messages=[
#                         {
#                             "role": "user",
#                             "content": prompt
#                         }
#                     ],
#                     temperature=1,
#                     max_tokens=1024,
#                     top_p=1,
#                     stream=False,
#                     stop=None
#                 )
#                 # Extract the response text
#                 response_text = response.choices[0].message.content.strip()
#                 processed_segments.append(response_text)
#             except Exception as e:
#                 print(f"Translation failed for segment: {segment}. Error: {e}")
#                 # If the translation fails, keep the original segment
#                 processed_segments.append(segment)
#         else:
#             # Keep emojis and special characters unchanged
#             processed_segments.append(segment)
    
#     # Join all segments back into a single sentence
#     final_text = ' '.join(processed_segments)
#     return final_text

def expand_contractions(text):
    return contractions.fix(text)

# def hinglish_to_english(tweet):
#     prompt = f"Translate '{tweet}' to English: and don't give explanation only give me the translated sentence as output "
#     response = client.chat.completions.create(
#         model="llama3-70b-8192",
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ],
#         temperature=1,
#         max_tokens=1024,
#         top_p=1,
#         stream=False,
#         stop=None
#     )

#     response_text = response.choices[0].message.content
#     # lines = response_text.split('\n')
#     # translated_sentence = lines[-1].strip()
#     return response_text


def build_subj_dicionary(lines):
    subj_dict = dict()
    for line in lines:
        splits = line.split(' ')
        if len(splits) == 6:
            word = splits[2][6:]        # the word analyzed
            word_type = splits[0][5:]   # weak or strong subjective
            pos = splits[3][5:]         # part of speech: noun, verb, adj, adv or anypos
            polarity = splits[5][14:]   # its polarity: can be positive, negative or neutral
            new_dict_entry = {pos: [word_type, polarity]}
            if word in subj_dict.keys():
                subj_dict[word].update(new_dict_entry)
            else:
                subj_dict[word] = new_dict_entry
    return subj_dict


def get_subj_lexicon():
    lexicon = utils.load_file(path + "/res/subjectivity_lexicon.tff")
    subj_dict = build_subj_dicionary(lexicon)
    return subj_dict


def get_emoji_dictionary():
    emojis = utils.load_file(path + "\\res\\emoji\\emoji_list.txt")
    emoji_dict = {}
    for line in emojis:
        line = line.split(" ", 1)
        emoji = line[0]
        description = line[1]
        emoji_dict[emoji] = description
    return emoji_dict


def build_emoji_sentiment_dictionary():
    new_emoji_sentiment_filename = path + "\\res\\emoji\\emoji_sentiment_dictionary.txt"
    if not os.path.exists(new_emoji_sentiment_filename):
        filename = path + "\\res\\emoji\\emoji_sentiment_raw.txt"
        emojis = utils.load_file(filename)[1:]
        lines = []
        for line in emojis:
            line = line.split(",")
            emoji = line[0]
            occurences = line[2]
            negative = float(line[4]) / float(occurences)
            neutral = float(line[5]) / float(occurences)
            positive = float(line[6]) / float(occurences)
            description = line[7]
            lines.append(str(emoji) + "\t" + str(negative) + "\t" + str(neutral)
                         + "\t" + str(positive) + "\t" + description.lower())
            utils.save_file(lines, new_emoji_sentiment_filename)
    emoji_sentiment_data = utils.load_file(new_emoji_sentiment_filename)
    emoji_sentiment_dict = {}
    for line in emoji_sentiment_data:
        line = line.split("\t")
        # Get emoji characteristics as a list [negative, neutral, positive, description]
        emoji_sentiment_dict[line[0]] = [line[1], line[2], line[3], line[4]]
    return emoji_sentiment_dict


# Extract each tweet's emojis - obv. it's just a brute force solution (so, it's slow) but works in ALL cases
def extract_emojis(tweets):
    emojis = []
    for tw in tweets:
        tw_emojis = []
        for word in tw:
            chars = list(word)
            for ch in chars:
                if ch in emoji.EMOJI_DATA:
                    tw_emojis.append(ch)
        emojis.append(' '.join(tw_emojis))
    return emojis


# Replace a contraction (coming from possessives, verbs, emphasis or just bad language) by its longer form
def replace_contracted_form(contracted_word, pos="V", dictionary="\\res\\word_list.txt"):
    long_form = []
    if "'" in contracted_word:
        # print("Found apostrophe in word: ", contracted_word, ' with pos: ', pos)
        split_words = contracted_word.split("'")
        check_if_in_dict = False
        # If the contraction is a nominal + verbal or a proper noun + verbal
        if pos == 'L' or pos == 'M':
            long_form.append(split_words[0])
            if split_words[1].lower() in contractions:
                long_form.extend(contractions[split_words[1].lower()].split())
        # If the contraction is a whole verb (like let's or isn't)
        elif pos in ['V', 'Y', 'O'] and contracted_word.lower() in contractions:
            long_form.extend(contractions[contracted_word.lower()].split())
        # If the contraction is proper noun with possessive or a nominal with a possessive or even a (proper) noun
        elif pos in ['S', 'Z', 'D', 'N', '^']:
            if contracted_word.lower() in contractions:
                long_form.extend(contractions[contracted_word.lower()].split())
            elif split_words[1].lower() == 's':
                long_form.append(split_words[0])
            elif contracted_word.lower() in contractions:
                long_form.extend(contractions[contracted_word.lower()].split())
            else:
                check_if_in_dict = True
        # Can skip ' which are just punctuation marks (usually used to emphasize or quote something)
        elif pos ==',':
            # print("Punctuation, nothing to replace.", split_words[0], ' -- ', split_words[1])
            return []
        # Never replace contractions in emojis or emoticons (will be translated later)
        elif pos == 'E':
            long_form.append(contracted_word)
        else:
            check_if_in_dict = True
        if check_if_in_dict:
            # Attempt to separate words which have been separated by ' by human error
            clean0 = re.findall("[a-zA-Z]+", split_words[0])
            clean1 = re.findall("[a-zA-Z]+", split_words[1])
            if clean0 != [] and clean0[0].lower() in dictionary and clean1 != [] and clean1[0].lower() in dictionary:
                # print("Cleaned to ", clean0, ', ', clean1)
                long_form.extend([clean0[0], clean1[0]])
            else:
                # print("Word couldn't be de-contracted!")
                long_form.append(contracted_word)
        return long_form
    else:
        return long_form.append(contracted_word)


# Cannot do lemmatization with NLTK without changing the case - which we don't want
# So lemmatize but remember if upper case or startign with upper letter
# This will be needed when performing CMU pos-tagging or when extracting pragmatic features
def correct_spelling_but_preserve_case(lemmatizer, word):
    corrected = lemmatizer.lemmatize(word.lower(), 'v')
    corrected = lemmatizer.lemmatize(corrected)
    if word.isupper():
        return corrected.upper()
    if word[0].isupper():
        return corrected[0].upper() + corrected[1:]
    return corrected


# Reduce the length of the pattern (if repeating characters are found)
def reduce_lengthening(word, dictionary):
    if word.lower() in dictionary or word.isnumeric():
        return word
    # Pattern for repeating character sequences of length 2 or greater
    pattern2 = re.compile(r"(.)\1{2,}")
    # Pattern for repeating character sequences of length 1 or greater
    pattern1 = re.compile(r"(.)\1{1,}")
    # Word obtained from stripping repeating sequences of length 2
    word2 = pattern2.sub(r"\1\1", word)
    # Word obtained from stripping repeating sequences of length 1
    word1 = pattern1.sub(r"\1", word)
    # print("Reduced length from ", word, " w2 -- ", word2, " w1 -- ", word1)
    if word1.lower() in dictionary:
        return word1
    else:
        return word2


# Translate emojis (or a group of emojis) into a list of descriptions
def process_emojis(word, emoji_dict, translate_emojis=True):
    processed = []
    chars = list(word)
    remaining = ""
    for c in chars:
        if c in emoji_dict.keys() or c in emoji.EMOJI_DATA:
            if remaining != "":
                processed.append(remaining)
                remaining = ""
            if translate_emojis:
                if c in emoji_dict:
                    processed.extend(emoji_dict[c][3].lower().split())
            else:
                processed.extend(c)
        else:
            remaining += c
    if remaining != "":
        processed.append(remaining)
    if processed != []:
        return ' '.join(processed)
    else:
        return word


# TODO: Numerals - sarcasm heavily relies on them so find a way to extract meaning behind numbers
# Attempt to clean each tweet and make it as grammatical as possible
def grammatical_clean(tweets, pos_tags, word_file, filename, translate_emojis=True, replace_slang=True, lowercase=False,
                      replace_user_mentions=True, translate_to_english=False):
    stopwords = get_stopwords_list()
    print(stopwords)
        

    if not os.path.exists(filename):
        dictionary = utils.load_file(word_file)
        emoji_dict = build_emoji_sentiment_dictionary()
        lemmatizer = WordNetLemmatizer()
        corrected_tweets = []
        
        for tweet in tweets:

            if translate_to_english:
                tweet = hinglish_to_english(tweet)

            tweet = expand_contractions(tweet)
            
            # Remove escape characters like \n, \t, etc.
            tweet = tweet.replace('\n', ' ').replace('\t', ' ').strip()
            
            # Remove extra spaces
            
            corrected_tweet = []
            
            for word in tweet.split():
                # Process case sensitivity
                if lowercase:
                    t = word.lower()
                else:
                    t = word
                
                # Never include #sarca* hashtags
                if t.lower().startswith('#sarca'):
                    continue
                
                # Never include URLs
                if 'http' in t:
                    continue
                
                # Replace specific user mentions with a general user name
                if replace_user_mentions and t.startswith('@'):
                    t = '@user'
                
                # Remove hashtags
                if t.startswith("#"):
                    t = t[1:]
                
                # Remove unnecessary hyphens
                if t.startswith('-') or t.endswith('-'):
                    t = re.sub(r'[-]', '', t)
                
                # Process emojis
                emoji_translation = process_emojis(t, emoji_dict, translate_emojis=translate_emojis)
                if emoji_translation != t:
                    corrected_tweet.append(emoji_translation)
                    continue
                
                # Translate emoticons to their description
                if translate_emojis and t.lower() in wikipedia_emoticons:
                    translated_emoticon = wikipedia_emoticons[t.lower()].split()
                    corrected_tweet.extend(translated_emoticon)
                    continue
                elif t.lower() in emotiocons_to_emojis:
                    translated_emoticon = emotiocons_to_emojis[t.lower()]
                    corrected_tweet.append(translated_emoticon)
                    continue
                
                
                # Replace slang with full form
                if replace_slang and t.lower() in slang.keys():
                    slang_translation = slang[t.lower()]
                    corrected_tweet.extend(slang_translation.split())
                    continue
                
                # Lemmatize and remove stopwords
                token = t.lower()
                if token not in stopwords:
                    filtered_token = lemmatizer.lemmatize(token, 'v')  # Verb lemmatization
                    filtered_token = lemmatizer.lemmatize(filtered_token)
                    if filtered_token not in stopwords:
                        corrected_tweet.append(filtered_token) # General lemmatization
                    
                
                
            
            # Remove any extra spaces after processing the tweet
            cleaned_tweet = ' '.join(corrected_tweet).strip()
            cleaned_tweet =  re.sub(r'\s{2,}', ' ', cleaned_tweet)# Remove extra spaces
            cleaned_tweet = re.sub(r'[^\w\s@#]', '', cleaned_tweet)  # Keep @, #, alphanumeric

            corrected_tweets.append(cleaned_tweet)
        
        # Save the cleaned tweets
        utils.save_file(corrected_tweets, filename)
        
        # Print comparisons (optional)
        for dirty, corrected in zip(tweets, corrected_tweets):
            print("Dirty:\t%s\nCleaned:\t%s" % (dirty, corrected))
    
    # Load and return the cleaned tweets from file
    cleaned_tweets = utils.load_file(filename)
    return cleaned_tweets

def clean(tweets):

    word_file=path+"\\res\\word_list.txt"
    stopwords = get_stopwords_list()
    dictionary = utils.load_file(word_file)
    emoji_dict = build_emoji_sentiment_dictionary()
    lemmatizer = WordNetLemmatizer()
    corrected_tweets = []
        
    for tweet in tweets:
        tweet = hinglish_to_english(tweet)
        #     Apply initial regex substitutions
        # translator = Translator()

        # tweet = translator.translate(tweet, dest='en').text
        # print(tweet)
        tweet = expand_contractions(tweet)
            
        # Remove escape characters like \n, \t, etc.
        tweet = tweet.replace('\n', ' ').replace('\t', ' ').strip()

        tweet = re.sub(r'\s{2,}', ' ', tweet)  # Replace multiple spaces with a single space
        corrected_tweet = []
            
        for word in tweet.split():
                # Process case sensitivity
            t = word.lower()
            
                # Never include #sarca* hashtags
            if t.lower().startswith('#sarca'):
                continue
                
                # Never include URLs
            if 'http' in t:
                continue
                
                # Replace specific user mentions with a general user name
            if  t.startswith('@'):
                 t = '@user'
                
                # Remove hashtags if specified
            if t.startswith("#"):
                t = t[1:]
                    
                # Remove unnecessary hyphens
            if t.startswith('-') or t.endswith('-'):
                    t = re.sub(r'[-]', '', t)
                
                # Process emojis
            emoji_translation = process_emojis(t, emoji_dict, translate_emojis=True)
            if emoji_translation != t:
                corrected_tweet.append(emoji_translation)
                continue
                
                    
                # Check and correct repeating characters
                
                # Translate emoticons to their description
            if t.lower() in wikipedia_emoticons:
                translated_emoticon = wikipedia_emoticons[t.lower()].split()
                corrected_tweet.extend(translated_emoticon)
                continue
            elif t.lower() in emotiocons_to_emojis:
                translated_emoticon = emotiocons_to_emojis[t.lower()]
                corrected_tweet.append(translated_emoticon)
                continue
                
                # Replace slang with full form
            if  t.lower() in slang.keys():
                slang_translation = slang[t.lower()]
                corrected_tweet.extend(slang_translation.split())
                continue
                
                # Lemmatize and remove stopwords
            token = t.lower()
            if token not in stopwords:
                filtered_token = lemmatizer.lemmatize(token, 'v')
                filtered_token = lemmatizer.lemmatize(filtered_token)
                if filtered_token not in stopwords:
                    corrected_tweet.append(filtered_token)

                
            
        
        cleaned_tweet = ' '.join(corrected_tweet).strip()
        cleaned_tweet =  re.sub(r'\s{2,}', ' ', cleaned_tweet)# Remove extra spaces
        cleaned_tweet = re.sub(r'[^\w\s@#]', '', cleaned_tweet)  # Keep @, #, alphanumeric

        corrected_tweets.append(cleaned_tweet)

    utils.save_file(corrected_tweets, "corrected_sample.txt")
        
        # Print comparisons (optional)
    for dirty, corrected in zip(tweets, corrected_tweets):
        print("Dirty:\t%s\nCleaned:\t%s" % (dirty, corrected))
    
    # Load and return the cleaned tweets from file
    cleaned_tweets = utils.load_file("corrected_sample.txt")
    return cleaned_tweets




def get_stopwords_list(filename="stopwords.txt"):
    stopwords = utils.load_file(path + "\\res\\" + filename)
    return stopwords


def get_grammatical_data(train_filename, test_filename, dict_filename,
                         translate_emojis=True, replace_slang=True, lowercase=True):
    # Load the train and test sets
    print("Loading data...")
    train_tokens = utils.load_file(path + "\\res\\tokens\\tokens_" + train_filename)
    train_pos = utils.load_file(path + "\\res\\pos\\pos_" + train_filename)
    test_tokens = utils.load_file(path + "\\res\\tokens\\tokens_" + test_filename)
    test_pos = utils.load_file(path + "\\res\\pos\\pos_" + test_filename)

    if translate_emojis and replace_slang and lowercase:
        save_path = path + "\\res\\data\\finest_grammatical_"
    else:
        save_path = path + "\\res\\data\\grammatical_"

    # Clean the data and brind it to the most *grammatical* form possible
    gramm_train = grammatical_clean(train_tokens, train_pos, path + "\\res\\" + dict_filename, save_path + train_filename,
                                    translate_emojis=translate_emojis, replace_slang=replace_slang, lowercase=lowercase)
    gramm_test = grammatical_clean(test_tokens, test_pos, path + "\\res\\" + dict_filename, save_path + test_filename,
                                   translate_emojis=translate_emojis, replace_slang=replace_slang, lowercase=lowercase)
    return gramm_train, gramm_test

def extract_lemmatized_tweet(tokens, pos, use_verbs=True, use_nouns=True, use_all=False):
    lemmatizer = WordNetLemmatizer()
    clean_data = []
    for index in range(len(tokens)):
        if use_verbs and pos[index] == 'V':
            clean_data.append(lemmatizer.lemmatize(tokens[index].lower(), 'v'))
        if use_nouns and pos[index] == 'N':
            clean_data.append(lemmatizer.lemmatize(tokens[index].lower()))
        if use_all:
            lemmatized_word = lemmatizer.lemmatize(tokens[index].lower(), 'v')
            word = lemmatizer.lemmatize(lemmatized_word)
            if pos[index] not in ['^', ',', '$', '&', '!', '#', '@']:
                clean_data.append(word)
    return clean_data


def get_dataset():
    data_path = path + "\\res\\" + "\\"
    train_tweets = utils.load_file(data_path +"\\data\\"+ "finest_grammatical_clean_original_train.txt")
    test_tweets = utils.load_file(data_path +"\\data\\"+"finest_grammatical_clean_original_test.txt")
    # train_tweets = utils.load_file(data_path +"\\tokens\\"+ "tokens_clean_original_train.txt")
    # test_tweets = utils.load_file(data_path +"\\tokens\\"+"tokens_clean_original_test.txt")
    
    train_pos = utils.load_file(data_path + "\\pos\\"+"pos_clean_original_train.txt")
    test_pos = utils.load_file(data_path + "\\pos\\"+"pos_clean_original_test.txt")
    train_labels = [int(l) for l in utils.load_file(data_path +"\\datasets\\"+ "labels_train.txt")]
    test_labels = [int(l) for l in utils.load_file(data_path + "\\datasets\\"+"labels_test.txt")]
    print("Size of the train set: ", len(train_labels))
    print("Size of the test set: ", len(test_labels))
    return train_tweets, train_pos, train_labels, test_tweets, test_pos, test_labels

def get_dataset_1():
    data_path = path + "\\res\\" + "\\"
    
    train_tweets = utils.load_file(data_path +"\\tokens\\"+ "tokens_clean_original_train.txt")
    test_tweets = utils.load_file(data_path +"\\tokens\\"+"tokens_clean_original_test.txt")
    
    train_pos = utils.load_file(data_path + "\\pos\\"+"pos_clean_original_train.txt")
    test_pos = utils.load_file(data_path + "\\pos\\"+"pos_clean_original_test.txt")
    train_labels = [int(l) for l in utils.load_file(data_path +"\\datasets\\"+ "labels_train.txt")]
    test_labels = [int(l) for l in utils.load_file(data_path + "\\datasets\\"+"labels_test.txt")]
    print("Size of the train set: ", len(train_labels))
    print("Size of the test set: ", len(test_labels))
    return train_tweets, train_pos, train_labels, test_tweets, test_pos, test_labels


def pos_tagging(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    # Return only the POS tags in a list
    return [tag for word, tag in pos_tags]

   


if __name__ == '__main__':
    print(path)

    train_filename = "clean_original_train.txt"
    test_filename = "clean_original_test.txt"
    
    gramm_train, gramm_test = get_grammatical_data(train_filename, test_filename, dict_filename,
                                                   translate_emojis=True, replace_slang=True, lowercase=True)
    

