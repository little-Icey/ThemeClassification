# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
# from scipy.misc import imread
from random import choice

def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return choice(["rgb(94,38,18)", "rgb(128,128,105)", "rgb(39,72,98)"])

def draw_cloud(word_freq, save_path):
    # mask = imread(mask_path)
    wc = WordCloud(font_path='data/loli.ttc',  # 设置字体
                   background_color="white",  # 背景颜色
                   max_words=500,  # 词云显示的最大词数
                   # mask=mask,  # 设置背景图片
                   max_font_size=80,  # 字体最大值
                   width=500,
                   height=350,
                   mode='RGBA',
                   colormap='pink'
                   # random_state=42,
                   )
    wc.generate_from_frequencies(word_freq)

    # show
    # image_colors = ImageColorGenerator(mask)
    plt.figure()
    # plt.imshow(wc.recolor(color_func=image_colors), interpolation='bilinear')
    plt.imshow(wc)
    plt.axis("off")
    wc.to_file(save_path)
    plt.show()
    return

def exp():
    for i in range(5):
        freq = pd.read_csv(f"output/topics/topic_word_freq{i}.csv", header=None, index_col=0)
        input_freq = freq[1].to_dict()
        draw_cloud(input_freq, f"output/img/topic{i}.png")