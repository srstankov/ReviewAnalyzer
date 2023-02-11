from nltk.corpus import wordnet

def get_pos_tag(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

import string

import nltk
from nltk.stem import WordNetLemmatizer

def remove_punctuation(text):
    formatted_text = "".join([p for p in text if p not in string.punctuation])
    return formatted_text

def tokenize(text):
    tokenized_text = [word for word in text.split(" ")]
    return tokenized_text

def get_stopwords():
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.remove('not')
    stopwords.remove('nor')
    stopwords.remove('no')
    return stopwords

def remove_stopwords(text):
    stopwords = get_stopwords()
    result = [w for w in text if w not in stopwords]
    return result

# def remove_special_chars(text):
#     text = text.replace(r"(http|@)\S+", "")
#     text = text.replace(r"::", ": :")
#     text = text.replace(r"’", "'")
#     text = text.replace(r"[^a-z\':_]", " ")
#     return text

def transform_short_forms(text):
    text = text.replace(r"can't", 'can not')
    text = text.replace(r"cannot", 'can not')
    text = text.replace(r"'m", ' am')
    text = text.replace(r"'re", ' are')
    text = text.replace(r"n't", ' not')
    text = text.replace(r"'ll", ' will')
    text = text.replace(r"'s", ' is')
    text = text.replace(r"'ve", ' have')
    return text

def lemmatize_text(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(text)
    lemmatized_text = [wordnet_lemmatizer.lemmatize(t[0], get_pos_tag(t[1])) for t in pos_tags]
    return lemmatized_text

def remove_digits(text):
    text = [w for w in text if all(not ch.isdigit() for ch in w)]
    return text

def remove_empty_strings(text):
    text = [s for s in text if len(s) > 0]
    return text

def clean_text(text):
    text = text.lower()
    text = transform_short_forms(text)
    text = remove_punctuation(text)
    text = tokenize(text)
    text = remove_digits(text)
    text = remove_stopwords(text)
    text = remove_empty_strings(text)
    text = lemmatize_text(text)
    return text

def clean_df(reviews_df):

    reviews_df["review_clean"] = reviews_df["reviewText"].apply(lambda a: clean_text(a))

    reviews_df["cleaned_text"] = [[' '.join(i)]for i in reviews_df["review_clean"]]

    return reviews_df
