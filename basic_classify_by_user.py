# -*- coding:utf-8 -*-
import jieba.posseg
import re
import os

jieba.load_STOP_WORDS = set([w.strip() for w in open("data/stopwords.txt", encoding='utf-8').readlines()])
STOP_WORDS = set([w.strip() for w in open("data/stopwords.txt", encoding='utf-8').readlines()])


class Users(object):
    def __init__(self, user_list):
        self.user_list = user_list

    def __iter__(self):
        for user in self.user_list:
            with open(user, encoding='utf-8') as f:
                user_text = f.read()
            yield cut_words_with_pos(user_text)

def split_by_user(filepath):
    user_list = []
    user_name_list = []
    with open('file_name.txt', encoding='utf-8') as f:
        while True:
            line = f.readline()
            # user_txt = 'data/userdata/'+ line[0:-1]
            # user_name = re.split(r'.', user_txt)
            # user_list.append(user_txt)
            # user_name_list.append(user_name)
            if not line:
                break
            user_txt = 'data/userdata/'+ line[0:-1]
            user_name = re.split(r'.', user_txt)
            user_list.append(user_txt)
            user_name_list.append(user_name)
    return user_list, user_name_list




def cut_words_with_pos(text):
    seg = jieba.posseg.cut(text)
    res = []
    for i in seg:
        if i.flag in ["a", "v", "x", "n", "an", "vn", "nz", "nt", "nr"] and is_fine_word(i.word):
            res.append(i.word)
    return list(res)

def file_test():
    with open('file_name.txt','w') as f:
        path = 'data/userdata/'
        files = os.listdir(path)
        files.sort()

        s = []
        for file in files:
            if not os.path.isdir(path + file):
                file_name = str(file)
                s.append(file_name)
                f.write(file_name+'\n')
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