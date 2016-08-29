#coding:utf-8

#stop wordの抽出

import stop_words as sw

stop_words = sw.get_stop_words('english')

def is_stopword(word):
    return word in stop_words

with open("./sentiment.txt", "r") as f:
    stop_words = sw.get_stop_words('english')

    for line in f:
        print(is_stopword(line))