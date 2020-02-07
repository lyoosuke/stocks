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
#ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier
#Numpyのimport
import numpy as np
#pyplotのimport
import matplotlib.pyplot as plt

#決定木の可視化
#from sklearn.tree import export_graphviz

name_data = pd.read_csv("name.csv", encoding="shift-jis")
count_name = len(name_data)

for name in range(200, 202):
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
    #print(rsi_13day)


    #解析銘柄の5日間のMACD
    macd_5day = []
    macd = []
    for i in range(4, count_d):
        sum5 = 0
        for s in range(0, 5):
            sum5 += float(deviation_stock5_13[i-s])
        macd_5day.append(float(sum5)/float(5))
    #print(macd_5day)
    for i in range(0, count_d-4):
            macd.append(float(deviation_stock5_13[i+4] - macd_5day[i]))
    #print(macd)

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

    for i in range(0, count_s-1-14):
        #ATRはTRの14日平均移動線
        sum_14 = 0
        for s in range(0, 14):
            sum_14 += tr[i+s]
        atr_14day.append(float(sum_14)/float(14))

    #以下はパラボリックを求めるプログラム
    #AF(Acceleration Factor)：加速因数
    AF = 0.02
    #EP(Extreme Point)：買い持ちしている期間の最高値、または餅売りしている期間の最安値
    #SAR(Stop and Reversal):パラボリックの値
    #AF:0.02~0.2の設定で行われることが多い。
    SAR = []
    temp = []#初期値を決めるためのtemp
    trend_frag = 0 #上昇トレンドなら1を下降トレンドなら0を代入する

    #以下は初期値を決めるプログラム
    trend = float(float(stock_data.loc[3, ['調整後終値']])-float(stock_data.loc[0, ['調整後終値']]))
    if trend > 0:
        trend_frag = 1
        #上昇トレンドであれば初期値を初日の5パーセント下がった状態に設定
        first = float(stock_data.loc[4, ['調整後終値']]) * 0.95
        for i in range(0, 4):
            temp.append(float(stock_data.loc[i, ['高値']]))
        EP = max(temp)
    else:
        trend_frag = 0
        first = float(stock_data.loc[4, ['調整後終値']]) * 1.05
        for i in range(0, 4):
            temp.append(float(stock_data.loc[i, ['安値']]))
        EP = min(temp)

    #5日目のパラボリックはSAR = 前日のSAR(first) + AF×(EP-前日のSAR(first))
    #5日目以降のパラボリックを特徴量とできるような計算を行う。
    SAR.append(first + AF *(EP - first))
    count = 0 #上昇トレンドや下降トレンドが何日間続いたかというのを記録する
    takane = [] #高値だけを入れるリスト
    yasune = [] #安値だけ入れるリスト
    Max = 0.0

    for i in range(0, count_s):
        takane.append(float(stock_data.loc[i, ['高値']]))
        yasune.append(float(stock_data.loc[i, ['安値']]))
    Min = max(takane)

    for i in range(5, count_s):
        if (trend_frag == 1) and SAR[i-5] < float(stock_data.loc[i, ['調整後終値']]):
            if EP < float(stock_data.loc[i, ['高値']]):
                EP = float(stock_data.loc[i, ['高値']])
                if AF < 0.2:        #AFの値は最大で0.2
                    AF += 0.02
                SAR.append(SAR[i-5] + AF *(EP - SAR[i-5]))
            else:
                SAR.append(SAR[i-5] + AF *(EP - SAR[i-5]))
            count += 1
        elif (trend_frag == 1) and SAR[i-5] >= float(stock_data.loc[i, ['調整後終値']]):
            AF = 0.02
            EP = float(stock_data.loc[i, ['安値']])
            for j in range(i-count-1, i):
                if Max < takane[j]:
                    Max = takane[j]
            SAR.append(Max)
            Max = 0.0
            trend_frag = 0
            count = 0
        elif (trend_frag == 0) and SAR[i-5] > float(stock_data.loc[i, ['調整後終値']]):
            if EP > float(stock_data.loc[i, ['安値']]):
                EP = float(stock_data.loc[i, ['安値']])
                if AF < 0.2:        #AFの値は最大で0.2
                    AF += 0.02
                SAR.append(SAR[i-5] + AF *(EP - SAR[i-5]))
            else:
                SAR.append(SAR[i-5] + AF *(EP - SAR[i-5]))
            count += 1
        elif (trend_frag == 0) and SAR[i-5] <= float(stock_data.loc[i, ['調整後終値']]):
            AF = 0.02
            EP = float(stock_data.loc[i, ['高値']])
            for j in range(i-count-1, i):
                if Min > yasune[j]:
                    Min = yasune[j]
            SAR.append(Min)
            Min = max(takane)
            trend_frag = 1
            count = 0



    #以下DMIを求めるプログラム
    #以下はDMの計算
    #+DM = 当日の高値ー前日の高値、−DM = 前日の安値ー当日の安値
    plusDM = []
    minusDM = []

    for i in range(0, count_s-1):
        DM1 = float(stock_data.loc[i+1, ['高値']]) - float(stock_data.loc[i, ['高値']])
        DM2 = float(stock_data.loc[i, ['安値']]) - float(stock_data.loc[i+1, ['安値']])
        if DM1 < 0:
            DM1 = 0
        elif DM2 < 0:
            DM2 = 0
        if DM1 >= DM2:
            DM2 = 0
        elif DM2 >= DM1:
            DM1 = 0
        plusDM.append(DM1)
        minusDM.append(DM2)
    #以下はsumの計算
    pDM_7day = []
    mDM_7day = []
    for i in range(0, count_s-1-7):
        sum_pDM = 0
        sum_mDM = 0
        for s in range(0, 7):
            sum_pDM += plusDM[i+s]
            sum_mDM += minusDM[i+s]
        pDM_7day.append(float(sum_pDM))
        mDM_7day.append(float(sum_mDM))


    #以下はTRの計算
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
    #以下はsumの計算
    tr_7day = []
    for i in range(0, count_s-1-7):
        #ATRはTRの14日平均移動線
        sum_7 = 0
        for s in range(0, 7):
            sum_7 += tr[i+s]
        tr_7day.append(float(sum_7))

    #以下はDIについての計算
    plusDI = []
    minusDI = []
    for i in range(0, len(tr_7day)):
        plusDI.append(pDM_7day[i] / tr_7day[i] * 100)
        minusDI.append(mDM_7day[i] / tr_7day[i] * 100)

    #以下はADX(Average Directional Index)についての計算
    DX = []
    for i in range(0, len(plusDI)):
        if((plusDI[i] + minusDI[i]) != 0):
            DX.append(abs(plusDI[i] - minusDI[i]) / (plusDI[i] + minusDI[i]) * 100)
        else:
            DX.append(0)
    ADX = []
    for i in range(0, len(DX)-7):
        sum_DX = 0
        flag_count = 0
        for s in range(0, 7):
            if (DX[i+s] == 0):
                flag_count += 1
            else:
                sum_DX += DX[i+s]
        ADX.append(float(sum_DX) / float(7-flag_count))


    #テクニカル分析のモメンタム
    momentum_10day = []
    #Momentumは当日の終値からn日前の終値を引くだけ
    for i in range(0, count_s-1-10):
        momentum_10day.append(float(float(stock_data.loc[i+10, ['調整後終値']])-float(stock_data.loc[i, ['調整後終値']])))



    #以下はストキャスティクスの計算を行う。
    #ストキャスティクスにはファーストとスローがある。
    FpK = [] #First % K
    FpD = [] #First % D

    for i in range(0, count_s-9):
        temp_max = 0
        temp_min = 100000
        for j in range(i, i+9):
            t_max = float(stock_data.loc[j, ['高値']])
            if temp_max < t_max:
                temp_max = t_max
            t_min = float(stock_data.loc[j, ['安値']])
            if temp_min > t_min:
                temp_min = t_min
        FpK.append(float(float((stock_data.loc[i+8, ['調整後終値']]) - temp_min) * 100) / float(temp_max - temp_min))

    for i in range(0, len(FpK)-3):
        temp = 0
        for j in range(0, 3):
            temp += float(FpK[i+j])
        FpD.append(temp / float(3))

    #次にスローキャスティクスを求める。
    SpD = []

    for i in range(0, len(FpD)-3):
        temp = 0
        for j in range(0, 3):
            temp += float(FpD[i+j])
        SpD.append(temp / float(3))



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
        successive_data.append([rsi_13day[i+1], rsi_13day[i+2], deviation_stock5_13[i+1], deviation_stock5_13[i+2], deviation_stock5_13[i+3],
        macd[i-1], atr_14day[i-1], atr_14day[i], atr_14day[i+1], momentum_10day[i+4], momentum_10day[i+5], momentum_10day[i+6],
        SAR[i+8], SAR[i+9], SAR[i+10], SAR[i+11], ADX[i], ADX[i+1], ADX[i+2], plusDI[i+6], plusDI[i+7], plusDI[i+8], minusDI[i+6],
        minusDI[i+7], minusDI[i+8],modified_data[i+12], FpK[i+7], FpD[i+5], SpD[i+3]])
        #上昇率が0以上なら1、そうでないなら0を格納
        if ((modified_data[i+15] + modified_data[i+16] + modified_data[i+17] + modified_data[i+18] + modified_data[i+19] / 5) > 0):
            answers.append(1)
        else:
            answers.append(0)

    #print(answers)
    #データの分割（データの80%を訓練用に、20％をテスト用に分割する）
    X_train, X_test, y_train, y_test =train_test_split(successive_data, answers, train_size=0.8,test_size=0.2,random_state=1)
    #print(X_train)


    clf = RandomForestClassifier(n_estimators=100, max_depth=17, random_state=0)
    clf = clf.fit(X_train, y_train)

    # 再学習後のモデルによるテスト
    # トレーニングデータを用いた予測
    y_train_pred = clf.predict(X_train)
    # テストデータを用いた予測
    y_val_pred = clf.predict(X_test)

    #print(y_val_pred)
    #print(len(y_val_pred))

    #print(y_test)
    #print(len(y_test))

    # 正解率の計算
    train_score = accuracy_score(y_train, y_train_pred)
    test_score = accuracy_score(y_test, y_val_pred)
    # 正解率を表示
    print("トレーニングデータに対する正解率：" + str(train_score * 100) + "%")
    print("テストデータに対する正解率：" + str(test_score * 100) + "%")

    #file = open('final_result.csv', 'a')
    #file.write(str(int(name_data.loc[name, ['銘柄']])) + "," + str(train_score) + "," + str(test_score) + '\n')
    #file.close()


"""
    #特徴量の重要度
feature = clf.feature_importances_

#特徴量の重要度を上から順に出力する
f = pd.DataFrame({'number': range(0, len(feature)),
             'feature': feature[:]})
f2 = f.sort_values('feature',ascending=False)
f3 = f2.loc[:, 'number']

#特徴量の名前
label = ["rsi_13day[i+1]", "rsi_13day[i+2]", "deviation_stock5_13[i+1]", "deviation_stock5_13[i+2]", "deviation_stock5_13[i+3]",
"macd[i-1]", "atr_14day[i-1]", "atr_14day[i]", "atr_14day[i+1]", "momentum_10day[i+4]", "momentum_10day[i+5]", "momentum_10day[i+6]",
"SAR[i+8]", "SAR[i+9]", "SAR[i+10]", "SAR[i+11]", "ADX[i]", "ADX[i+1]", "ADX[i+2]", "plusDI[i+6]", "plusDI[i+7]", "plusDI[i+8]", "minusDI[i+6]",
"minusDI[i+7]", "minusDI[i+8]", "modified_data[i+12]", "FpK[i+7]", "FpD[i+5]", "SpD[i+3]"]

#特徴量の重要度順（降順）
indices = np.argsort(feature)[::-1]

for i in range(len(feature)):
    print(str(i + 1) + "   " + str(label[indices[i]]) + "   " + str(feature[indices[i]]))

plt.title('Feature Importance')
plt.bar(range(len(feature)),feature[indices], color='lightblue', align='center')
plt.xticks(range(len(feature)), label[indices], rotation=90)
plt.xlim([-1, len(feature)])
plt.tight_layout()
plt.show()
"""
