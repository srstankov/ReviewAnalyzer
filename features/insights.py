from collections import Counter
import matplotlib.pyplot as mplt
import wordcloud
import pandas as pd
from preprocessing import clean_data
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn

def extract_ngrams(tokens, ngram_size = 2):
    ngrams = zip(*[tokens[i:] for i in range(ngram_size)])
    ngrams_list = (" ".join(ng) for ng in ngrams)
    return ngrams_list

def unite_review_tokens_from_df(reviews_df, sentiment_tag = "None"):
    if sentiment_tag == "None":
        sentiment_reviews_df = reviews_df
    elif sentiment_tag == "Positive":
        sentiment_reviews_df = reviews_df[reviews_df["sentiment"] == "Positive"]
    else: #sentiment_tag == "Negative"
        sentiment_reviews_df = reviews_df[reviews_df["sentiment"] == "Negative"]

    united_review_tokens = [token for review_tokens in sentiment_reviews_df["review_clean"] for token in review_tokens]
    return united_review_tokens

def calc_word_frequencies(words, ngram_size = 1, output_top_words_length = 9):
    if ngram_size > 1:
        words = extract_ngrams(words, ngram_size)
    freq_counter = Counter(words)
    frequency_list = freq_counter.most_common(output_top_words_length)
    return frequency_list

def create_work_cloud(text_or_counter):
    stop_words = clean_data.get_stopwords()
    if isinstance(text_or_counter, str):
        cloud = wordcloud.WordCloud(stopwords=stop_words).generate(text_or_counter)
    else:
        text_or_counter = Counter(word for word in text_or_counter if word not in stop_words)
        cloud = wordcloud.WordCloud(stopwords=stop_words).generate_from_frequencies(text_or_counter)
    mplt.imshow(cloud)
    mplt.axis("off")
    mplt.show()

def calc_dataset_sentiment_ratio(reviews_df):
    sentiment_ratio = reviews_df["sentiment"].value_counts(normalize=True)
    return sentiment_ratio

def plot_sentiment_distribution(reviews_df):
    pd.value_counts(reviews_df['sentiment']).plot(kind = "bar")
    mplt.show()

def plot_ratings_distribution(reviews_df):
    pd.value_counts(reviews_df['rating']).plot(kind="bar")
    mplt.show()

def extract_common_words(reviews_df, sentiment_tag = "Positive", n = 10, ngram_size = 1):
    words = unite_review_tokens_from_df(reviews_df[reviews_df['sentiment'] == sentiment_tag])
    if ngram_size > 1:
        words = extract_ngrams(words, ngram_size)
    most_common_words = Counter(words).most_common(n)
    return dict(most_common_words)

def dummy(doc):
    return doc

def extract_features(reviews_df, ngram_rng = (2,2)):
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', tokenizer=dummy, preprocessor=dummy, token_pattern=None,
                                       ngram_range=ngram_rng)
    features = tfidf_vectorizer.fit_transform(reviews_df["review_clean"])
    return tfidf_vectorizer.get_feature_names_out()