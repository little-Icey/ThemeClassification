# -*- coding:utf-8 -*-
import jieba.posseg
import re
import os

jieba.load_STOP_WORDS = set([w.strip() for w in open("data/stopwords.txt", encoding='utf-8').readlines()])
STOP_WORDS = set([w.strip() for w in open("data/stopwords.txt", encoding='utf-8').readlines()])


class Users(object):
    def __init__(self, user_list, user_name, user_gender, user_age):
        self.user_list = user_list
        self.user_name = user_name
        self.user_gender = user_gender
        self.user_age = user_age

    def __iter__(self):
        for user, gender, age in zip(self.user_list, self.user_gender, self.user_age):
            with open(user, encoding='utf-8') as f:
                user_text = f.read()
            print(f"{self.user_list.index(user)} is ok")
            yield cut_words_with_pos(user_text), gender, age


def split_by_user(filepath):
    user_list = []
    user_name_list = []
    user_gender_list = []
    user_age_list = []
    with open(filepath, encoding='utf-8') as f:
        while True:
            line = f.readline()
            # user_txt = 'data/userdata/'+ line[0:-1]
            # user_name = re.split(r'.', user_txt)
            # user_list.append(user_txt)
            # user_name_list.append(user_name)
            if not line:
                break
            user_txt = 'data/userdata/' + line[0:-1]
            user_name = line[0:-1].split('.')[0][:-2]
            user_gender = line[0:-1].split('.')[0][-2:-1]
            user_age = line[0:-1].split('.')[0][-1]

            user_list.append(user_txt)
            user_name_list.append(user_name)
            user_gender_list.append(user_gender)
            user_age_list.append(user_age)
    return user_list, user_name_list, user_gender_list, user_age_list


def cut_words_with_pos(text):
    seg = jieba.posseg.cut(text)
    res = []
    for i in seg:
        if i.flag in ["a", "v", "x", "n", "an", "vn", "nz", "nt", "nr"] and is_fine_word(i.word):
            res.append(i.word)
    return list(res)


def file_test():
    with open('file_name_male.txt', 'w') as f:
        path = 'data/userdata/'
        files = os.listdir(path)
        files.sort()

        # s = []
        # for file in files:
        #     if (not os.path.isdir(path + file)) and file[32] == 'F':
        #         file_name = str(file)
        #         # s.append(file_name)
        #         f.write(file_name + '\n')

        for i in range(56000):
            if files[i][32] == 'M':
                f.write(files[i] + '\n')
    # print(s)


# 过滤词长，过滤停用词，只保留中文
def is_fine_word(word, min_length=2):
    rule = re.compile(r"^[\u4e00-\u9fa5]+$")
    if len(word) >= min_length and word not in STOP_WORDS and re.search(rule, word):
        return True
    else:
        return False


if __name__ == '__main__':
    file_test()
    # split_by_user('file_name.txt')