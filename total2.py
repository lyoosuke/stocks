# train_test_splitのimport
from sklearn.model_selection import train_test_split
# accuracy_scoreのimport
from sklearn.metrics import accuracy_score
# Pandasのimport
import pandas as pd
# グリッドサーチのimport
from sklearn.model_selection import GridSearchCV
#ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier
#Numpyのimport
import numpy as np
#pyplotのimport
import matplotlib.pyplot as plt
#勾配ブースティング
from sklearn.ensemble import GradientBoostingClassifier
# サポートベクターマシーンのimport
from sklearn import svm
from sklearn.svm import SVC
# グリッドサーチのimport
from sklearn.model_selection import GridSearchCV
#標準化のライブラリ
from sklearn import preprocessing
import statistics
import csv
#特徴量選択ライブラリ
from sklearn.feature_selection import SelectKBest, f_regression

name_data = pd.read_csv("name.csv", encoding="shift-jis")
count_name = len(name_data)

successive_data = []
successive_data_n = []
answers = []
answers_n = []

#(["rsi_13day[i+1]", "rsi_13day[i+2]", "deviation_stock[i+2]", "deviation_stock[i+3]", "macd[i-1]",
#"atr[i+1]", "momentum[i+6]", "momentum_1ago[i+4]", "momentum_1ago[i+5]", "parabolic[i+10]",
#"parabolic[i+11]", "DMI[i+2]", "modified_data[i+13]", "modified_data[i+14]", "stochastics[i+2]"])
#feature_list = [0]
feature_list = [i for i in range(0, 15)]

with open("successive2.csv", "r") as file:
    reader = csv.reader(file, lineterminator="\n")
    for row in reader:
        temp = []
        for i in feature_list:
            temp.append(float(row[i]))
        successive_data.append(temp)
with open("answers2.csv", "r") as file:
    reader = csv.reader(file, lineterminator="\n")
    for row in reader:
        answers.append(float(row[0]))
with open("test2.csv", "r") as file:
    reader = csv.reader(file, lineterminator="\n")
    for row in reader:
        temp = []
        for i in feature_list:
            temp.append(float(row[i]))
        successive_data_n.append(temp)
    #print(successive_data_n)
with open("test-ans2.csv", "r") as file:
    reader = csv.reader(file, lineterminator="\n")
    for row in reader:
        answers_n.append(float(row[0]))

#以下はランダムフォレストによって予測を行う。
for s in range(30, 31):
    for t in range(3, 4):
        clf = RandomForestClassifier(n_estimators=s*10, max_depth=t, random_state=1)
        #clf = SVC(kernel='linear', random_state=None)
        clf = clf.fit(successive_data, answers)
        #print(answers)
        # トレーニングデータを用いた予測
        y_train_pred = clf.predict(successive_data)
        #print(y_train_pred[len(y_train_pred)-1000:])
        #print(y_train_pred)
        # 正解率の計算
        train_score = accuracy_score(answers, y_train_pred)
        print("RandomForest(" + str(s*10)+","+ str(t)+")トレーニングデータに対する正解率：" + str(train_score * 100) + "%")
        sum_t = 0
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for name in range(0, count_name):
            #各銘柄それぞれのデータ
            meigara = []
            #それの答え
            meigara_answer = []
            for i in range(226 * name, 226 * (name+1)):
                meigara.append(successive_data_n[i])
                meigara_answer.append(answers_n[i])
            #print(meigara)
            y_val_pred = clf.predict(meigara)
            #モデルが当たった日
            #collect_a = []
            #モデルが外れた日
            #wrong_a = []
            for i in range(0, len(y_val_pred)):
                if(meigara_answer[i]==1 and y_val_pred[i]==1):
                    #wrong_a.append(i)
                    TP += 1
                elif(meigara_answer[i]==0 and y_val_pred[i]==1):
                    FP += 1
                    #collect_a.append(i)
                elif(meigara_answer[i]==1 and y_val_pred[i]==0):
                    FN += 1
                else:
                    TN += 1
            #print("モデルが当たった日:")
            #print(collect_a)
            #print("モデルが外れた日:")
            #print(wrong_a)
            # 正解率の計算
            test_score = accuracy_score(meigara_answer, y_val_pred)
            sum_t += test_score
            # 正解率を表示
            #print("RandomForest(" + str(s*10)+","+ str(t) +","+ str(count_name+1)+"銘柄)テストデータに対する正解率：" + str(test_score * 100) + "%")
            #file = open('tree_result6.csv', 'a')
            #file.write(str(int(name_data.loc[name, ['銘柄']])) + "," + str(test_score) + '\n')
            #file.close()

            #特徴量の重要度
            #feature = clf.feature_importances_

            #特徴量の重要度を上から順に出力する
            #f = pd.DataFrame({'number': range(0, len(feature)),
                         #'feature': feature[:]})
            #f2 = f.sort_values('feature',ascending=False)
            #f3 = f2.loc[:, 'number']

            #特徴量の名前
            #label = ["rsi_13day[i+1]", "rsi_13day[i+2]",
            #"macd[i-1]", "atr[i+1]", "momentum[i+6]", "momentum_1ago[i+4]", "parabolic[i+10]", "parabolic[i+11]",
            #"DMI[i+2]", "modified_data[i+13]", "modified_data[i+14]", "stochastics[i+2]"]

            #特徴量の重要度順（降順）
            #indices = np.argsort(feature)[::-1]

        #for i in range(len(feature)):
            #print(str(i + 1) + "   " + str(label[indices[i]]) + "   " + str(feature[indices[i]]))

        #print("TP = " + str(TP))
        #print("FP = " + str(FP))
        #print("FN = " + str(FN))
        #print("TN = " + str(TN))
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f_score = (2 * recall * precision) / (recall + precision)
        print(f_score)
        ave = float(sum_t)/226
        print("ave:" + str(ave))

"""
#以下は勾配ブースティングによって分析を行う。
gbrt = GradientBoostingClassifier(learning_rate=0.15, n_estimators=200, random_state=0)
gbrt.fit(successive_data, answers)
# トレーニングデータを用いた予測
y_train_pred = gbrt.predict(successive_data)
print(y_train_pred)
# 正解率の計算
train_score = accuracy_score(answers, y_train_pred)
print("boostingトレーニングデータに対する正解率：" + str(train_score * 100) + "%")
sum_t = 0
for name in range(0, count_name):
    #各銘柄それぞれのデータ
    meigara = []
    #それの答え
    meigara_answer = []
    for i in range(226 * name, 226 * (name+1)):
        meigara.append(successive_data_n[i])
        meigara_answer.append(answers_n[i])

    y_val_pred = gbrt.predict(meigara)

    #モデルが当たった日
    collect_a = []
    #モデルが外れた日
    wrong_a = []
    for i in range(0, len(y_val_pred)):
        if(meigara_answer[i] != y_val_pred[i]):
            wrong_a.append(i)
        else:
            collect_a.append(i)
    #print("モデルが当たった日:")
    #print(collect_a)
    #print("モデルが外れた日:")
    #print(wrong_a)
    # 正解率の計算
    test_score = accuracy_score(meigara_answer, y_val_pred)
    sum_t += test_score
    # 正解率を表示
    #print("boostingテストデータに対する正解率：" + str(test_score * 100) + "%")
    #file = open('tree_result7.csv', 'a')
    #file.write(str(int(name_data.loc[name, ['銘柄']])) + "," + str(test_score) + '\n')
    #file.close()

    #特徴量の重要度
    #feature = gbrt.feature_importances_

    #特徴量の重要度を上から順に出力する
    #f = pd.DataFrame({'number': range(0, len(feature)),
                 #'feature': feature[:]})
    #f2 = f.sort_values('feature',ascending=False)
    #f3 = f2.loc[:, 'number']

    #特徴量の名前
    #label = ["rsi_13day[i+1]", "rsi_13day[i+2]",
    #"macd[i-1]", "atr[i+1]", "momentum[i+6]", "momentum_1ago[i+4]", "parabolic[i+10]", "parabolic[i+11]",
    #"DMI[i+2]", "modified_data[i+13]", "modified_data[i+14]", "stochastics[i+2]"]

    #特徴量の重要度順（降順）
    #indices = np.argsort(feature)[::-1]

#for i in range(len(feature)):
    #print(str(i + 1) + "   " + str(label[indices[i]]) + "   " + str(feature[indices[i]]))

ave = float(sum_t)/226
print("ave:" + str(ave))
"""
