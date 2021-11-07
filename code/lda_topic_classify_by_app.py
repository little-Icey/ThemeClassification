#coding=utf-8
import numpy as np
from basic_classify_by_app import *
from sklearn.feature_extraction.text import CountVectorizer
import lda
import matplotlib.pyplot as plt
import seaborn as sns

def get_lda_input(chapters):
    corpus = [" ".join(word_list) for word_list in chapters]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X.toarray(), vectorizer

def lda_train(weight, vectorizer):
    model = lda.LDA(n_topics=300, n_iter=1000, random_state=1)
    model.fit(weight)

    doc_num = len(weight)
    topic_word = model.topic_word_
    vocab = vectorizer.get_feature_names()
    titles = ["App{}".format(i) for i in range(1, doc_num+1)]

    n_top_words = 20
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    doc_topic = model.doc_topic_
    print(doc_topic, type(doc_topic))
    # plot_topic(doc_topic)
    for i in range(doc_num):
        print("{} (top topic: {})".format(titles[i], np.argsort(doc_topic[i])[:-4:-1]))

def main():
    app_list = split_by_app("data/userdata/00a6fe5e801047991e6d85e2b976bdff.txt")
    apps = MyApps(app_list)
    weight, vectorizer = get_lda_input(apps)
    lda_train(weight, vectorizer)

if __name__ == '__main__':
    main()