# サポートベクターマシーンのimport
from sklearn import svm
# train_test_splitのimport
from sklearn.model_selection import train_test_split
# accuracy_scoreのimport
from sklearn.metrics import accuracy_score
# Pandasのimport
import pandas as pd
# グリッドサーチのimport
from sklearn.model_selection import GridSearchCV
#決定木
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (roc_curve, auc, accuracy_score)
#決定木の可視化
from sklearn.tree import export_graphviz

name_data = pd.read_csv("name.csv", encoding="shift-jis")
count_name = len(name_data)

for name in range(210, count_name):
    stock_data = pd.read_csv(str(int(name_data.loc[name, ['銘柄']])) + "_tech4.csv", encoding="shift-jis")
    #print(stock_data)
    # 要素数の設定
    count_s = len(stock_data)

    #解析銘柄の13日平均移動線を求める
    average_13day = []
    for i in range(12,count_s-1):
        sum13 = 0
        for s in range(0, 13):
            sum13 += float(stock_data.loc[i-s, ['調整後終値']])
        average_13day.append(float(sum13)/float(13))

    #解析銘柄の5日平均移動線を求める
    average_5day = []
    for i in range(12,count_s-1):
        sum5 = 0
        for s in range(0, 5):
            sum5 += float(stock_data.loc[i-s, ['調整後終値']])
        average_5day.append(float(sum5)/float(5))

    #解析銘柄の5日-13日乖離率の計算
    deviation_stock5_13 = []
    for i in range(0,count_s-13):
        deviation_stock5_13.append(float((float(average_5day[i]) - float(average_13day[i]))/float(average_13day[i])))
    #print(deviation_stock3_8)

    count_d = len(deviation_stock5_13)

    #解析銘柄の乖離率が5%を越えるとシグナル
    sign_5per = []
    for i in range(0, count_d):
        if deviation_stock5_13[i] < -0.05:
            sign_5per.append(1)
        elif deviation_stock5_13[i] > 0.05:
            sign_5per.append(-1)
        else:
            sign_5per.append(0)

    #print(sign_5per)

    #解析銘柄の13日RSIを求める
    rsi_13day = []
    for i in range(0, count_s-12):
        up_sum = 0
        down_sum = 0
        for j in range(1, 13):
            up = float(stock_data.loc[i+j, ['調整後終値']] - stock_data.loc[i+j-1, ['調整後終値']])
            if up >= 0:
                up_sum += up
            else:
                down_sum -= up
        rsi = (up_sum/13*100) / ((up_sum/13)+(down_sum/13)) - 50
        rsi_13day.append(rsi)
    print(rsi_13day)

    #解析銘柄の5日間のMACD
    macd_5day = []
    macd = []
    for i in range(4, count_d):
        sum5 = 0
        for s in range(0, 5):
            sum5 += float(deviation_stock5_13[i-s])
        macd_5day.append(float(sum5)/float(5))
        macd_5day.append(float(deviation_stock5_13[i-4] + deviation_stock5_13[i-3] +deviation_stock5_13[i-2] +deviation_stock5_13[i-1] +deviation_stock5_13[i])/ float(5))
    print(macd_5day)
    for i in range(0, count_d-4):
        macd.append(float(deviation_stock5_13[i+4] - macd_5day[i])/float(macd_5day[i]))
    print(macd)

    #テクニカル分析のATR
    atr_14day = []
    #TRはTrueRangeの略、tr1~tr3の中で最大のものの値を入れる
    tr = []
    for i in range(0, count_s-1-1):
        #tr1は当日高値ー当日安値、tr2は当日高値ー前日終値、tr3は前日終値ー当日安値
        tr1 = float(float(stock_data.loc[i+1, ['高値']])-float(stock_data.loc[i+1, ['安値']]))
        tr2 = float(float(stock_data.loc[i+1, ['高値']])-float(stock_data.loc[i, ['終値']]))
        tr3 = float(float(stock_data.loc[i, ['終値']])-float(stock_data.loc[i+1, ['安値']]))
        if(tr1 > tr2 and tr1 > tr3):
            tr.append(tr1)
        elif(tr2 > tr3):
            tr.append(tr2)
        else:
            tr.append(tr3)
    #print(tr)

    for i in range(0, count_s-1-14):
        #ARTはTRの14日平均移動線
        sum_14 = 0
        for s in range(0, 14):
            sum_14 += tr[i+s]
        atr_14day.append(float(sum_14)/float(14))


    #テクニカル分析のモメンタム
    momentum_10day = []
    #Momentumは当日の終値からn日前の終値を引くだけ
    for i in range(0, count_s-1-10):
        momentum_10day.append(float(float(stock_data.loc[i+10, ['調整後終値']])-float(stock_data.loc[i, ['調整後終値']])))



    # 株価の上昇率を算出、おおよそ-1.0～1.0の範囲に収まるように調整
    modified_data = []

    for i in range(1,count_s):
        modified_data.append((float(stock_data.loc[i, ['調整後終値']])*100 / float(stock_data.loc[i-1, ['調整後終値']]))-100)
    print(modified_data)
    # 要素数の設定
    count_m = len(modified_data)


    # 説明変数となるデータを格納するリスト
    successive_data = []
    # 正解値を格納するリスト　価格上昇: 1 価格低下:0
    answers = []

    #  説明変数となるデータを格納していく
    for i in range(1, count_s-21):
        successive_data.append([rsi_13day[i+2], deviation_stock5_13[i+2], deviation_stock5_13[i+3], macd[i-1], sign_5per[i+3], atr_14day[i+1], momentum_10day[i+6]])
        #上昇率が0以上なら1、そうでないなら0を格納
        if ((modified_data[i+15] + modified_data[i+16] + modified_data[i+17] + modified_data[i+18] + modified_data[i+19] / 5) > 0):
            answers.append(1)
        else:
            answers.append(0)

    print(answers)
    #データの分割（データの80%を訓練用に、20％をテスト用に分割する）
    X_train, X_test, y_train, y_test =train_test_split(successive_data, answers, train_size=0.8,test_size=0.2,random_state=1)
    #print(X_train)


    clf = DecisionTreeClassifier(max_depth=17, random_state=0)
    clf = clf.fit(X_train, y_train)

    # 再学習後のモデルによるテスト
    # トレーニングデータを用いた予測
    y_train_pred = clf.predict(X_train)
    # テストデータを用いた予測
    y_val_pred = clf.predict(X_test)

    print(y_val_pred)
    print(len(y_val_pred))

    print(y_test)
    print(len(y_test))

    # 正解率の計算
    train_score = accuracy_score(y_train, y_train_pred)
    test_score = accuracy_score(y_test, y_val_pred)
    # 正解率を表示
    print("トレーニングデータに対する正解率：" + str(train_score * 100) + "%")
    print("テストデータに対する正解率：" + str(test_score * 100) + "%")


    file = open( 'tree_result.csv', 'a')
    file.write(str(int(name_data.loc[name, ['銘柄']])) + "," + str(train_score) + "," + str(test_score) + '\n')
    file.close()
