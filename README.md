## 爬虫部分
使用爬虫爬取应用商城的APP简介

## LDA部分
根据用户APP列表得到用户APP描述文本，将这些文本作为LDA训练的语料库
使用LDA算法提取用户特征，设置15个主题，每个主题有20个主题词，训练1000轮
得到用户频率矩阵，每个用户的特征为15维，每一维代表该用户属于某一主题的概率

## MLP部分
使用多层感知机来进行分类，使用LDA部分得到的用户特征对性别作二分类
