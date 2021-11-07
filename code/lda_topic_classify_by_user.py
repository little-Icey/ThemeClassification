# coding=utf-8
import numpy as np
from basic_classify_by_user import *
from sklearn.feature_extraction.text import CountVectorizer
import lda
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import word_cloud


def top_topics(lt):
    d = {}
    for i in lt:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:5]


def get_lda_input(users):
    corpus = [" ".join(user_info[0]) for user_info in users]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X.toarray(), vectorizer


def lda_train(weight, vectorizer, gender):
    model = lda.LDA(n_topics=15, n_iter=1000, random_state=1)
    model.fit(weight)

    doc_num = len(weight)
    print("doc_num:{}".format(doc_num))
    topic_word = model.topic_word_  # 主题词的频率
    vocab = vectorizer.get_feature_names()
    titles = ["User{}".format(i) for i in range(1, doc_num + 1)]

    n_top_words = 20
    # 主题词
    topic_words_array, topic_words_freq_array = [], []
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        topic_words_array.append(topic_words)  # 主题词矩阵
        topic_words_freq_array.append(np.sort(topic_dist)[:-(n_top_words + 1):-1])  # 主题词频率矩阵
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        print(f"[*] {np.sort(topic_dist)[:-(n_top_words + 1):-1]}")  # 主题词频率

    # 提取出的主题 每个用户属于15个主题中某一个的概率
    doc_topic = model.doc_topic_
    print(doc_topic, len(doc_topic), type(doc_topic))
    filepath = 'output/numpy_data/data_numpy{}.csv'.format(gender)
    np.savetxt(filepath, doc_topic, delimiter=',')

    # top5的主题
    top_topic_array = []
    # plot_topic(doc_topic)
    for i in range(doc_num):
        print("{} (top topic: {})".format(titles[i], np.argsort(doc_topic[i])[:-6:-1]))  # argsort将元素从小到大排列，并提取出其索引
        top_topic_array.append(np.argsort(doc_topic[i])[:-6:-1])

    top_topic_array = np.asarray(top_topic_array)
    print(f"[*] top 5 topic:{top_topics(top_topic_array.flatten().tolist())}")

    return topic_words_array, topic_words_freq_array, [x[0] for x in top_topics(top_topic_array.flatten().tolist())]


def main():
    user_list, user_name_list = split_by_user("file_name.txt")
    users = Users(user_list)
    weight, vectorizer = get_lda_input(users)
    return lda_train(weight, vectorizer)


def generate_word_cloud():
    user_list, user_name_list, user_gender_list, user_age_list = split_by_user("file_name_male.txt")
    users = Users(user_list, user_name_list, user_gender_list, user_age_list)

    gender = input('[*] input gender>>> ')
    age = input('[*] input age>>> ')
    print(f"[+] generate word cloud for {gender}")

    weight, vectorizer = get_lda_input(users)
    words, freqs, top_themes = lda_train(weight, vectorizer, gender)

    # for i in top_themes:
    #     with open(f"output/topics/{gender}/topic_word_freq{top_themes.index(i)}.csv", 'w', encoding='utf-8') as f:
    #         for j in range(30):
    #             f.write(f"{words[i][j]}, {freqs[i][j]}\n")
    #         f.write('\n')
    # word_cloud.exp(gender)


if __name__ == '__main__':
    # main()
    generate_word_cloud()
