#!/usr/bin/env python
# coding: utf-8

# In[3]:


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils


# In[19]:


#MNISTのデータを読み込む
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#データをfloat32型に変換して正規化する
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float')
print(X_train)
print(y_train)
X_train /= 255
X_test /= 255
#ラベルデータを0-9までのカテゴリを表す配列に変換
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
print(X_train)
print(y_train)


# In[3]:


#モデルの構造を定義
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))


# In[14]:


#モデルを構築
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy'])
#データで訓練
hist = model.fit(X_train, y_train, epochs = 5)

#テストデータを用いて評価する
score = model.evaluate(X_test, y_test, verbose=1)
print('loss=', score[0])
print('accuracy=', score[1])


# In[23]:


import random

#BMIを計算して体型を返す
def calc_bmi(h, w):
    bmi = w / (h / 100) ** 2
    if bmi < 18.5: return "thin"
    if bmi < 25: return "normal"
    return "fat"

#出力ファイルの準備
fp = open("bmi.csv","w",encoding="utf-8")
fp.write("height,weight,label\r\n")

#ランダムなデータを生成
cnt = {"thin":0, "normal":0, "fat":0}
for i in range(20000):
    h = random.randint(120, 200)
    w = random.randint(35, 80)
    label = calc_bmi(h, w)
    cnt[label] += 1
    fp.write("{0},{1},{2}\r\n".format(h, w, label))
fp.close()
print("ok", cnt)


# In[24]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
import pandas as pd, numpy as np


# In[29]:


#BMIのデータを読み込んで正規化する
csv = pd.read_csv('bmi.csv')
print(csv)
#体重と身長のデータ
csv["weight"] /= 100
csv["height"] /= 200
X = csv[["weight", "height"]].as_matrix()
#ラベル
bclass = {"thin":[1,0,0], "normal":[0,1,0], "fat":[0,0,1]}
y = np.empty((20000, 3))
for i, v in enumerate(csv["label"]):
    y[i] = bclass[v]
    
#訓練データとテストデータを分ける
X_train, y_train = X[1:15001], y[1:15001]
X_test, y_test = X[15001:20001], y[15001:20001]


# In[26]:


#モデルの構造を定義
model = Sequential()
model.add(Dense(512, input_shape=(2,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(3))
model.add(Activation('softmax'))


# In[28]:


#モデルを構築
model.compile(
    loss='categorical_crossentropy',
    optimizer="rmsprop",
    metrics=['accuracy'])
#データで訓練
hist = model.fit(
    X_train, y_train,
    batch_size = 100,
    nb_epoch = 20,
    validation_split = 0.1,
    callbacks = [EarlyStopping(monitor='val_loss', patience=2)],
    verbose = 1)

#テストデータを用いて評価する
score = model.evaluate(X_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])


# In[36]:


import tensorflow as tf 
#sess = tf.compat.v1.Session()
image = []
for i in range(30, 2432):
    png = tf.io.read_file('9613_average_30_' + str(i) + '.png')
    image.append(tf.image.decode_png(png, channels=1, dtype=tf.uint8))
image_float = tf.cast(image,dtype=tf.float32)
print(image_float)
image_reshape = tf.reshape(image_float, [-1, 307200])
print(image_reshape)


# In[68]:


X_train = image_reshape[0:2001]
X_test = image_reshape[2001:2403]
X_train /= 255
X_test /= 255


# In[69]:


Y = pd.read_csv('modified_answer_9613.csv', encoding="shift-jis")
print(Y)
y_train = Y[0:2001]
y_test = Y[2001:2403]
y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)
print(y_train)
print(y_test)


# In[76]:


#モデルの構造を定義
model = Sequential()
model.add(Dense(1024, input_shape=(307200,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(2))
model.add(Activation('softmax'))


# In[77]:


#モデルを構築
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy'])
#データで訓練
hist = model.fit(X_train, y_train, steps_per_epoch=3, epochs=10)

#テストデータを用いて評価する
score = model.evaluate(x = X_test, y = y_test, steps=5)
print('loss=', score[0]/5)
print('accuracy=', score[1]/5)


# In[78]:


img_pred = model.predict(X_test, steps=5)
print(img_pred)


# In[ ]:




