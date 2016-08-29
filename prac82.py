#coding:utf-8

import stop_words as sw
import re

def is_stopword(word):
    return word in stop_words

stop_words = sw.get_stop_words('english')
sent = []
words = ""

with open("./sentiment.txt", "r") as f:
    stop_words = sw.get_stop_words('english')

    for line in f:
        for word in re.compile(r'[,.:;\s]').sub(" ",line).split():
            if(not(is_stopword(word))):
                #sentenses.append(word)
                words = words + " " + word

        sent.append(words)
        words = ""

print(sent)
