#!/usr/bin/env python
# coding: utf-8

# In[94]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_finance
import seaborn as sns
stock_data = pd.read_csv("1321_tech4.csv", encoding="shift-jis")
print(stock_data)
print(stock_data.shape)
count_s = len(stock_data)
rangeday = 30
x = np.arange(0,rangeday, 1)
print(x)
y = stock_data.loc[(count_s-rangeday):, ['終値']]
plt.plot(x, y)


# In[98]:


#解析銘柄の13日平均移動線を求める
average_13day = []
for i in range(12,count_s-1):
    sum13 = 0
    for s in range(0, 13):
        sum13 += float(stock_data.loc[i-s, ['調整後終値']])
    average_13day.append(float(sum13)/float(13))
y2 = average_13day[(len(average_13day)-rangeday):]
#解析銘柄の5日平均移動線を求める
average_5day = []
for i in range(12,count_s-1):
    sum5 = 0
    for s in range(0, 5):
        sum5 += float(stock_data.loc[i-s, ['調整後終値']])
    average_5day.append(float(sum5)/float(5))
y3 = average_5day[(len(average_5day)-rangeday):]
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
fig, ax= plt.subplots()
ax.plot(x, y, color='green')
ax.plot(x, y2, color='black')
ax.plot(x, y3, color='black')
for t in range(0, len(x)-1):
    x2 = np.arange(t, t+1.01, 0.01) 
    m_y21 = (y2[t+1]-y2[t])
    b_y21 = -m_y21*x[t] + y2[t]
    m_y31 = (y3[t+1]-y3[t])
    b_y31 = -m_y31*x[t] + y3[t]
    y21 = m_y21*x2 + b_y21
    y31 = m_y31*x2 + b_y31
    ax.fill_between(x2, y21, y31, where=y21 <=y31, facecolor='red', alpha=0.8)
    ax.fill_between(x2, y21, y31, where=y21 >y31, facecolor='blue', alpha=0.8)


# In[23]:


#RSIの作成
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
y4 = rsi_13day[len(rsi_13day)-rangeday:]
plt.plot(x, y4)


# In[24]:


#MACDの作成
deviation_stock5_13 = []
for i in range(0,count_s-13):
    deviation_stock5_13.append(float((float(average_5day[i]) - float(average_13day[i]))/float(average_13day[i])))

macd_5day = []
macd = []
count_d = len(deviation_stock5_13)
for i in range(4, count_d):
    sum5 = 0
    for s in range(0, 5):
        sum5 += float(deviation_stock5_13[i-s])
    macd_5day.append(float(sum5)/float(5))
    macd_5day.append(float(deviation_stock5_13[i-4] + deviation_stock5_13[i-3] +deviation_stock5_13[i-2] +deviation_stock5_13[i-1] +deviation_stock5_13[i])/ float(5))
for i in range(0, count_d-4):
    macd.append(float(deviation_stock5_13[i+4] - macd_5day[i])/float(macd_5day[i]))
y5 = macd[len(macd)-rangeday:]
plt.plot(x, y5)


# In[25]:


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
y6 = atr_14day[len(atr_14day)-rangeday:]
plt.plot(x, y6)


# In[26]:


#テクニカル分析のモメンタム
momentum_10day = []
#Momentumは当日の終値からn日前の終値を引くだけ
for i in range(0, count_s-1-10):
    momentum_10day.append(float(float(stock_data.loc[i+10, ['調整後終値']])-float(stock_data.loc[i, ['調整後終値']])))
y7 = momentum_10day[len(momentum_10day)-rangeday:]
plt.plot(x, y7)


# In[27]:


# 株価の上昇率を算出、おおよそ-1.0～1.0の範囲に収まるように調整
modified_data = []

for i in range(1,count_s):
    modified_data.append((float(stock_data.loc[i, ['調整後終値']])*100 / float(stock_data.loc[i-1, ['調整後終値']]))-100)
y8 = modified_data[len(modified_data)-rangeday:]
plt.plot(x, y8)


# In[28]:


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
y9 = SAR[len(SAR)-rangeday:]
plt.plot(x, y9)


# In[48]:


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
y10 = plusDI[len(plusDI)-rangeday:]
y11 = minusDI[len(minusDI)-rangeday:]
y12 = ADX[len(ADX)-rangeday:]
plt.plot(x, y10, color='blue')
plt.plot(x, y11, color='red')
plt.plot(x, y12, color='green')


# In[30]:


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
    
y13 = FpK[len(FpK)-rangeday:]
y14 = SpD[len(SpD)-rangeday:]
plt.plot(x, y13, color='blue')
plt.plot(x, y14, color='red')


# In[92]:


x = [0, 1, 2, 3, 4, 5, 6]
y1 = [1.3, 2.5, 2.3, 4.8, 1.3, 0.4, 5.0] 
y2 = [3.4, 5.7, 1.5, 4.4, 6.7, 3.3, 3.1]

fig, ax = plt.subplots()
ax.plot(x, y1, x, y2, color='black')      
for t in range(0, len(x)-1):
    x2 = np.arange(t, t+1.01, 0.01) 
    m_y11 = (y1[t+1]-y1[t])
    b_y11 = -m_y11*x[t] + y1[t]
    m_y21 = (y2[t+1]-y2[t])
    b_y21 = -m_y21*x[t] + y2[t]
    y11 = m_y11*x2 + b_y11
    y21 = m_y21*x2 + b_y21
    ax.fill_between(x2, y11, y21, where=y21 >y11, facecolor='yellow', alpha=0.5)
    ax.fill_between(x2, y11, y21, where=y21 <=y11, facecolor='red', alpha=0.5)
ax.set_title('Fill Between')
plt.show()

