#coding:utf-8

import random

sentences = []

with open("./rt-polaritydata/pos.txt", "r") as f:
    for line in f:
        sentences.append("+1 " + line)
        #sentences.append({'label':1, 'sentence':line})

with open("./rt-polaritydata/neg.txt", "r") as f:
    for line in f:
        sentences.append("-1 " + line)
        #sentences.append({'label':-1, 'sentence':line})


random.shuffle(sentences)

with open("./sentiment.txt", "w") as f:
    for s in sentences:
        f.write(s)

#for s in sentences:
#    print(s)
