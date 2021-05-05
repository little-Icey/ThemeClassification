# -*- coding:utf-8 -*-
import jieba.posseg
import re

jieba.load_STOP_WORDS = set([w.strip() for w in open("data/stopwords.txt", encoding='utf-8').readlines()])
STOP_WORDS = set([w.strip() for w in open("data/stopwords.txt", encoding='utf-8').readlines()])


# 将用户applist按应用划分
class MyApps(object):
    def __init__(self, app_list):
        self.app_list = app_list

    def __iter__(self):
        for app in self.app_list:
            yield cut_words_with_pos(app)


# 按app分割
def split_by_app(filepath):
    applist = []
    with open('data/userdata/00a6fe5e801047991e6d85e2b976bdff.txt', encoding='utf-8') as f:
        while True:
            app_name = f.readline()
            app_desc = f.readline()
            applist.append(app_name + app_desc)
            if not app_name:
                break
    return applist


def cut_words_with_pos(text):
    seg = jieba.posseg.cut(text)
    res = []
    for i in seg:
        if i.flag in ["a", "v", "x", "n", "an", "vn", "nz", "nt", "nr"] and is_fine_word(i.word):
            res.append(i.word)
    return list(res)

# 过滤词长，过滤停用词，只保留中文
def is_fine_word(word, min_length=2):
    rule = re.compile(r"^[\u4e00-\u9fa5]+$")
    if len(word) >= min_length and word not in STOP_WORDS and re.search(rule, word):
        return True
    else:
        return False
