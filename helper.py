import re
from bs4 import BeautifulSoup
import distance
from fuzzywuzzy import fuzz
import pickle
import numpy as np
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

word2vec = pickle.load(open('word2vec_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
xgboost_model = pickle.load(open('xgb_model.pkl', 'rb'))


def test_common_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1 & w2)

def test_total_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return (len(w1) + len(w2))


def test_fetch_token_features(q1, q2):
    SAFE_DIV = 0.0001

    STOP = nltk.corpus.stopwords.words('english')
    STOP_WORDS = set(STOP)

    token_features = [0.0] * 8

    # Converting the Sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    # Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))

    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))

    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])

    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features


def test_fetch_length_features(q1, q2):
    length_features = [0.0] * 3

    # Converting the Sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    # Absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))

    # Average Token Length of both Questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)

    return length_features


def test_fetch_fuzzy_features(q1, q2):
    fuzzy_features = [0.0] * 4

    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features


def preprocess(q):
    q = str(q).lower().strip()

    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')

    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')

    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    # Removing HTML tags
    q = BeautifulSoup(q, features="lxml")
    q = q.get_text()

    # Remove punctuations
    pattern = re.compile(r'\W')
    q = re.sub(pattern, ' ', q).strip()

    return q

# Define a function to tokenize and lemmatize text
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

def tokenize_and_lemmatize(q1, q2):
    lemmatizer = WordNetLemmatizer()
    # Tokenize and lemmatize both questions
    tokens_q1 = word_tokenize(q1.lower())
    tokens_q2 = word_tokenize(q2.lower())
    lemmatized_q1 = [lemmatizer.lemmatize(token) for token in tokens_q1 if token.isalpha()]
    lemmatized_q2 = [lemmatizer.lemmatize(token) for token in tokens_q2 if token.isalpha()]
    # Return cleaned strings for both questions
    return ' '.join(lemmatized_q1), ' '.join(lemmatized_q2)

def prepare_training_dataset(cleaned_questions, final_df, y, vector_size=150):
    # Train Word2Vec on all sentences
    sentences = [q.split() for q in cleaned_questions]
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=5, min_count=1, workers=1, sg=1, seed=42)
    wv = model.wv

    # Split cleaned_questions into question1 and question2
    half = len(cleaned_questions) // 2
    q1 = cleaned_questions[:half]
    q2 = cleaned_questions[half:]

    def avg_w2v(text):
        words = text.split()
        vectors = [wv[w] for w in words if w in wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

    # Compute embeddings for both question parts
    q1_vecs = np.array([avg_w2v(q) for q in q1], dtype=np.float32)
    q2_vecs = np.array([avg_w2v(q) for q in q2], dtype=np.float32)

    return q1_vecs, q2_vecs

def query_point_creator(q1, q2, wv=None, vector_size=150, scaler=None):
    input_query = []

    # preprocess
    q1_clean = preprocess(q1)
    q2_clean = preprocess(q2)

    # fetch basic features (to match notebook)
    input_query.append(len(q1_clean))  # q1_len
    input_query.append(len(q2_clean))  # q2_len
    input_query.append(len(q1_clean.split(" ")))  # q1_num_words
    input_query.append(len(q2_clean.split(" ")))  # q2_num_words
    input_query.append(test_common_words(q1_clean, q2_clean))  # common_word_count
    # word_share: common_word_count / (q1_num_words + q2_num_words)
    q1_num_words = len(q1_clean.split(" "))
    q2_num_words = len(q2_clean.split(" "))
    input_query.append(round(test_common_words(q1_clean, q2_clean) / (q1_num_words + q2_num_words + 1e-6), 2))

    # fetch token features
    token_features = test_fetch_token_features(q1_clean, q2_clean)
    input_query.extend(token_features)

    # fetch length based features
    length_features = test_fetch_length_features(q1_clean, q2_clean)
    input_query.extend(length_features)

    # fetch fuzzy features
    fuzzy_features = test_fetch_fuzzy_features(q1_clean, q2_clean)
    input_query.extend(fuzzy_features)

    # --- SCALE THE 21 ENGINEERED FEATURES ---
    features = np.array(input_query).reshape(1, -1)
    if scaler is not None:
        features = scaler.transform(features)
    features = features.flatten()

    # tokenize and lemmatize for embeddings
    lemmatized_q1, lemmatized_q2 = tokenize_and_lemmatize(q1_clean, q2_clean)

    # word2vec embeddings (if wv is provided)
    def avg_w2v(text):
        if wv is None:
            return np.zeros(vector_size)
        words = text.split()
        vectors = [wv[w] for w in words if w in wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(vector_size)

    q1_vec = avg_w2v(lemmatized_q1)
    q2_vec = avg_w2v(lemmatized_q2)

    # concatenate all features and embeddings
    return np.hstack((features, q1_vec, q2_vec))