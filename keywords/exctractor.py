from preprocessing import clean_data
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def sort_matrix(coo_matrix):
    matrix_tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(matrix_tuples, key=lambda a: (a[1], a[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_vectors, topn=5):
    sorted_vectors = sorted_vectors[:topn]
    score_values = []
    feature_values = []
    for i, score in sorted_vectors:
        score_values.append(round(score, 3))
        feature_values.append(feature_names[i])
    results = {}
    for i in range(len(feature_values)):
        results[feature_values[i]] = score_values[i]

    return results


def get_keywords(vectorizer, feature_names, review):

    vectorized_review = vectorizer.transform([review])

    sorted_vectors = sort_matrix(vectorized_review.tocoo())

    keywords = extract_topn_from_vector(feature_names, sorted_vectors)

    return list(keywords.keys())

def dummy(doc):
    return doc

def exctract_keywords_from_df(reviews_df,ngram_rng = (1,3)):
    reviews = reviews_df["review_clean"]
    vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True, analyzer='word', tokenizer=dummy, preprocessor=dummy,
                                token_pattern=None, ngram_range=ngram_rng)
    vectorizer.fit_transform(reviews[10::])
    features = vectorizer.get_feature_names_out()
    result = []
    for rev in reviews[0:10]:
        keywords_df = {}
        keywords_df['review_text'] = reviews_df.loc[reviews_df['review_clean'].apply(lambda x: x == rev), "reviewText"]
        keywords_df['top_keywords'] = get_keywords(vectorizer, features, rev)
        result.append(keywords_df)

    return pd.DataFrame(result)

def exctract_review_keywords(review_text, reviews_df, ngram_rng = (1,3)):
    review_cleaned = clean_data.clean_text(review_text)
    reviews = reviews_df["review_clean"]
    vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True, analyzer='word', tokenizer=dummy, preprocessor=dummy,
                                 token_pattern=None, ngram_range=ngram_rng)
    vectorizer.fit_transform(reviews)
    features = vectorizer.get_feature_names_out()
    result = []
    keywords_df = {}
    keywords_df['review_text'] = review_text
    keywords_df['top_keywords'] = get_keywords(vectorizer, features, review_cleaned)
    result.append(keywords_df)

    return pd.DataFrame(result)