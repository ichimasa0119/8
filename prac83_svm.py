#coding:utf-8

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

from nltk import stem
import re
import random

#センテンスを取得
def get_sentence():
    sentences = []

    with open("./rt-polaritydata/pos.txt", "r") as f:
        for line in f:
            sentences.append({'label':1, 'sentence':line})

    with open("./rt-polaritydata/neg.txt", "r") as f:
        for line in f:
            sentences.append({'label':-1, 'sentence':line})

    random.shuffle(sentences)

    return sentences

def is_stopword(word):
    return word in count_vectorizer.get_stop_words()


def create_feature(sentence):
    fuature = []
    words = ""
    stemmer = stem.PorterStemmer()
    sentences = []
    labels = []

    # 素性の作成
    for s in sentence:
        for word in re.compile(r'[,.:;\s]').sub(" ", s['sentence']).split():
            if (not (is_stopword(word))):
                words = words + " " + stemmer.stem(word)
        sentences.append(words)
        labels.append(s['label'])
        words = ""
    fuature.append(sentences)
    fuature.append(labels)
    return fuature

def create_words_weight(vacabulary, coef):
    word_weight = []
    for vocab, weight in zip(vacabulary, coef):
        word_weight.append({'vocab': vocab, 'weight': weight})

    return word_weight

def export_predict(X, Y, lr):
    true_labels = Y
    pre_labels = lr.predict(X)
    prob = lr.predict_proba(X)

    with open("./prob.txt", "w") as f:
        for i, t_label in enumerate(true_labels):
            if t_label==1:
                line = str(t_label) + "  " + str(pre_labels[i]) + "  "+ str(prob[i][1]) +"\n"
            elif t_label==-1:
                line = str(t_label) + "  " + str(pre_labels[i]) + "  "+ str(prob[i][0]) +"\n"

            f.writelines(line)

if __name__ == "__main__":
    #目的変数(体調情報)
    Y = [1,2,3]
    #説明変数

    count_vectorizer = CountVectorizer(stop_words='english')

    sentence = get_sentence()
    fuature = create_feature(sentence)

    print("fuature:  ",fuature[0])

    #単語数のカウント
    feature_vectors = count_vectorizer.fit_transform(fuature[0])
    vocabulary = count_vectorizer.get_feature_names()

    X = feature_vectors.toarray()
    Y = fuature[1]

    print("vocabulary",vocabulary)
    print("feature_vectors.toarray",feature_vectors.toarray())
    print(Y)

    #学習
    lr = LogisticRegression(C=1000.0)
    lr.fit(X, Y)

    print(lr.get_params)

    #文章を与えて確率を予測
    prob = lr.predict_proba(X[0].reshape(1, -1))[0]
    print(prob)

    #重み print(lr.coef_)
    #バイアス print(lr.intercept_)

    print("~~~~~~~~~~~~~~~~~~~~~~~")
    #単語と重みの紐付け
    word_weight = create_words_weight(vocabulary, lr.coef_[0])

    #TOP10を抽出する
    descend_weight = sorted(word_weight, key=lambda x: x["weight"],reverse=True)
    #UNDER10を抽出
    ascend_weight = sorted(word_weight, key=lambda x: x["weight"])
    print("top10", descend_weight[0:11])
    print("under10", ascend_weight[0:11])

    y_pre = lr.predict(X)

    #ラベルと確率をファイル出力
    export_predict(X, Y, lr)

    '''
    予測の正解率，正例に関する適合率，再現率，F1スコア
    精度(適合率, precision)：正と予測したデータのうち，実際に正であるものの割合
    再現率 (recall)：実際に正であるもののうち，正であると予測されたものの割
    F値 (F尺度, F-measure)：精度と再現率の調和平均．
    '''
    #正解率
    print(accuracy_score(Y, y_pre))
    print(classification_report(Y, y_pre))

    '''
    5-分割交差検定
    データを5分割して4つを訓練データ、1つをテストデータとして用いる手法
    '''
    '''
    scores = cross_validation.cross_val_score(lr, X, Y, cv=5, scoring='accuracy')
    print("accuracy:  ", scores)
    scores = cross_validation.cross_val_score(lr, X, Y, cv=5, scoring='precision')
    print("precision:  ", scores)
    scores = cross_validation.cross_val_score(lr, X, Y, cv=5, scoring='recall')
    print("recall:  ", scores)
    scores = cross_validation.cross_val_score(lr, X, Y, cv=5, scoring='f1_weighted')
    print("f1_weighted:  ", scores)
    '''

    #thresholds = [{ 1:0.99, -1:0.01 },{ 1:0.5, -1:0.5 },{ 1:0.01, -1:0.99 }]
    #t = [0.99, 0.5, 0.01]

    thresholds = [{1:0.99, -1:0.01},{1:0.95, -1:0.05},{1:0.9, -1:0.1},{1:0.85, -1:0.15},{1:0.8, -1:0.2}
        ,{1:0.75, -1:0.25},{1:0.7, -1:0.3},{1:0.65, -1:0.35},{1:0.6, -1:0.4},{1:0.55, -1:0.45},{1:0.5, -1:0.5}
        ,{1:0.45, -1:0.55},{1:0.4, -1:0.6}, {1:0.35, -1:0.7}, {1:0.3, -1:0.7}, {1:0.25, -1:0.8},{1:0.2, -1:0.8}
        , {1:0.15, -1:0.85},{1:0.1, -1:0.9}, {1:0.05, -1:0.95}, {1:0.01, -1: 0.99}]
    t = [0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.01]

    precision_rates = []
    recall_rates = []
    #thresholds = [t / 20 for t in range(10)]

    for threshold in thresholds:
        scores = cross_validation.cross_val_score(lr, X, Y, cv=5, scoring='precision')
        precision_rates.append(scores[0])
        scores = cross_validation.cross_val_score(lr, X, Y, cv=5, scoring='recall')
        recall_rates.append(scores[0])

    print(precision_rates)
    print(recall_rates)

    plt.plot(thresholds, precision_rates, label="precision", color="red")
    plt.plot(thresholds, recall_rates, label="recall", color="blue")

    plt.xlabel("threshold")
    plt.ylabel("rate")
    plt.xlim(-0.05, 0.5)
    plt.ylim(0, 1)
    plt.title("Logistic Regression")
    plt.show()
