import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import clean_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

def load_dataset(rows_start = 0, step = 12000):
    reviews_df = pd.read_csv(r'C:\Users\SGComp\PycharmProjects\ReviewAnalyzer\dataset\all_kindle_review.csv')
    reviews_df = reviews_df[rows_start: rows_start + step]
    reviews_df = clean_data.clean_df(reviews_df)
    reviews_df['sentiment'] = numpy.where(reviews_df["rating"] > 3.0, "Positive", "Negative")
    # reviews_df['sentiment'] = numpy.where(reviews_df["rating"] == 3, "Neutral", reviews_df['sentiment'])
    reviews_df = reviews_df[reviews_df["rating"] != 3]
    return reviews_df

def dummy(doc):
    return doc

def predict_sentiment_on_review(review, reviews_df, ngram_rng = (1,3)):
    x_train, x_test, y_train, y_test = train_test_split(reviews_df["review_clean"], reviews_df['sentiment'],
                                                        test_size=0.2)
    vectorizer = TfidfVectorizer(analyzer='word', tokenizer=dummy, preprocessor=dummy, token_pattern=None, ngram_range=ngram_rng)
    x_train_vectorized = vectorizer.fit_transform(x_train)

    x_test_vectorized = vectorizer.transform(x_test)
    svm_model = svm.SVC(kernel="linear")
    svm_model.fit(x_train_vectorized, y_train)
    y_pred = svm_model.predict(x_test_vectorized)

    review_clean = clean_data.clean_text(review)

    review_clean = [review_clean]
    review_vectorized = vectorizer.transform(review_clean)

    sentiment = svm_model.predict(review_vectorized)[0]

    return sentiment

def evaluate_model_on_df(reviews_df, ngram_rng = (1,3)):
    reviews_df = reviews_df.sample(frac = 1)
    x_train, x_test, y_train, y_test = train_test_split(reviews_df["review_clean"], reviews_df['sentiment'],
                                                        test_size=0.25)
    vectorizer = TfidfVectorizer(analyzer='word', tokenizer=dummy, preprocessor=dummy, token_pattern=None, ngram_range=ngram_rng)
    x_train_vectorized = vectorizer.fit_transform(x_train)

    x_test_vectorized = vectorizer.transform(x_test)
    svm_model = svm.SVC(class_weight= "balanced", kernel="linear")
    svm_model.fit(x_train_vectorized, y_train)
    y_pred = svm_model.predict(x_test_vectorized)

    f1 = f1_score(y_test, y_pred, average="macro").round(3)
    accuracy = accuracy_score(y_test, y_pred).round(3)
    print("F1 score: ", f1)
    print("Accuracy: ", accuracy)
    # svm_score = svm_model.score(x_test_vectorized, y_test)
    # print("SVM Score: ", svm_score)
