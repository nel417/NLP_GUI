import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import json
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

file_name = "books2.json"

en_core_web = spacy.load("en_core_web_sm")
en_core_web.add_pipe('spacytextblob')


class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"


class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE


class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews

    def get_text(self):
        return [x.text for x in self.reviews]

    def get_labels(self):
        return [x.sentiment for x in self.reviews]

    def even(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
        #         print(len(negative))
        #         print(len(positive))
        shrink_positive = positive[:len(negative)]
        self.reviews = negative + shrink_positive
        random.shuffle(self.reviews)


reviews = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review["reviewText"], review["overall"]))


def sentiment_from_scratch(t):
    Train, Test = train_test_split(reviews, test_size=0.33, random_state=42)
    container_train = ReviewContainer(Train)
    container_test = ReviewContainer(Test)
    container_train.even()
    x_train = container_train.get_text()
    y_train = container_train.get_labels()
    container_test.even()
    x_test = container_test.get_text()
    y_test = container_test.get_labels()
    vec = TfidfVectorizer()
    train_x_vec = vec.fit_transform(x_train)
    test_x_vec = vec.transform(x_test)
    y_train.count(Sentiment.NEGATIVE)
    classify = svm.SVC(kernel='linear')
    classify.fit(train_x_vec, y_train)
    classify.predict(test_x_vec[0])
    classify_dec = DecisionTreeClassifier()
    from sklearn.metrics import f1_score
    classify_dec.fit(train_x_vec, y_train)
    classify_dec.predict(test_x_vec[8])
    new_test = vec.transform([t])
    output = classify.predict(new_test)
    print(classify.score(test_x_vec, y_test))
    print(classify_dec.score(test_x_vec, y_test))
    return_score = f1_score(y_test,
                            classify.predict(test_x_vec),
                            average=None,
                            labels=[Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE])

    return output, return_score


def get_text_characteristics(sentence):
    document = en_core_web(sentence)
    output_array = []
    for token in document:
        output = {
            "Index": token.i, "Token": token.text, "Tag": token.tag_, "POS": token.pos_,
            "Dependency": token.dep_, "Lemma": token.lemma_, "Shape": token.shape_,
            "Alpha": token.is_alpha, "Is Stop Word": token.is_stop
        }
        output_array.append(output)
    return {"output": output_array}


def get_entity(sentence):
    document = en_core_web(sentence)
    output_array = []
    for token in document.ents:
        output = {
            "Text": token.text, "Start Char": token.start_char,
            "End Char": token.end_char, "Label": token.label_
        }
        output_array.append(output)
    return {"output": output_array}


def get_text_sentiment(sentence):
    document = en_core_web(sentence)

    url_sent_score = []
    url_sent_label = []
    total_pos = []
    total_neg = []
    sentiment = document._.blob.polarity
    sentiment = round(sentiment, 2)

    if sentiment > 0:
        sent_label = "Positive"
    else:
        sent_label = "Negative"

    url_sent_label.append(sent_label)
    url_sent_score.append(sentiment)
    positive_words = []
    negative_words = []

    for x in document._.blob.sentiment_assessments.assessments:
        if x[1] > 0:
            positive_words.append(x[0][0])
        elif x[1] < 0:
            negative_words.append(x[0][0])
        else:
            pass

    total_pos.append(', '.join(set(positive_words)))
    total_neg.append(', '.join(set(negative_words)))

    output = {"Score": url_sent_score, "Label": url_sent_label,
              "Positive words": total_pos, "Negative Words": total_neg}

    return {"output": output}
