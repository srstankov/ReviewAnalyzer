from sentiment_analyzer import predict_sentiment
from features import insights
from keywords import exctractor
from preprocessing import clean_data

reviews_df = predict_sentiment.load_dataset(rows_start=6000, step = 1000)
predict_sentiment.evaluate_model_on_df(reviews_df, ngram_rng=(1,3))

review_pos = "Incredible story. I really like it."
review_neg = "Horrible book, the plot is very boring and the action is very slow. I can't recommend it for sure."
sentiment_pos = predict_sentiment.predict_sentiment_on_review(review_pos, reviews_df)
print(sentiment_pos)
sentiment_neg = predict_sentiment.predict_sentiment_on_review(review_neg, reviews_df)
print(sentiment_neg)

united_review_tokens = insights.unite_review_tokens_from_df(reviews_df)

word_frequencies = insights.calc_word_frequencies(united_review_tokens, ngram_size=2)
print("Most common word frequencies:")
print(word_frequencies)

sentiment_ratio = insights.calc_dataset_sentiment_ratio(reviews_df)
print("Sentiment ratio:")
print(sentiment_ratio)

positive_common_words = insights.extract_common_words(reviews_df, "Positive", ngram_size=2)
negative_common_words = insights.extract_common_words(reviews_df, "Negative", ngram_size=2)
print("Positive common words:")
print(positive_common_words)
print("Negative common words:")
print(negative_common_words)

# print("Number of positive reviews:")
# print(len(reviews_df[reviews_df['sentiment'] == "Positive"]))

keywords_df = exctractor.exctract_keywords_from_df(reviews_df)
print(keywords_df)

neg_review_keywords = exctractor.exctract_review_keywords(review_neg, reviews_df)
print("Negative review keywords:")
print(neg_review_keywords)

# insights.plot_sentiment_distribution(reviews_df)
# insights.plot_ratings_distribution(reviews_df)

united_positive_tokens = insights.unite_review_tokens_from_df(reviews_df, "Positive")
united_negative_tokens = insights.unite_review_tokens_from_df(reviews_df, "Negative")

# insights.create_work_cloud(united_review_tokens)
# insights.create_work_cloud(united_positive_tokens)
# insights.create_work_cloud(united_negative_tokens)
