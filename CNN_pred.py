import tensorflow as tf
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
#sess = tf.compat.v1.Session()
kobetu_list = ["4502", "4506", "4689", "4902", "6752", "6773", "7004", "8028", "9613", "9735"]
check_meigara = "4502"
image = []
X_train = []
X_test = []
y_train = []
y_test = []
for kobetu in kobetu_list:
    for i in range(30, 2432):
        png = tf.io.read_file(kobetu + '_average_10_' + str(i) + '.png')
        image.append(tf.image.decode_png(png, channels=1, dtype=tf.uint8))
    image_float = tf.cast(image,dtype=tf.float32)
    #print(image_float)
    image_reshape = tf.reshape(image_float, [-1, 3072])
    #print(image_reshape)
    X1 = image_reshape[0:2175]/255
    X2 = image_reshape[2175:2402]/255
    if kobetu == check_meigara:
        X_train = X1
        X_test = X2
    #else:
        #X_train = tf.concat([X_train, X1], 0)
        #X_test = tf.concat([X_test, X2], 0)

    #X_train.append(image_reshape[0:2175]/255)
    #X_test.append(image_reshape[2175:2403]/255)
    #X_train /= 255
    #X_test /= 255

    print(X_train)
    print(X_test)

    stock_data = pd.read_csv(kobetu + "_tech4.csv", encoding="shift-jis")
    Y = []
    answers = []
    for i in range(30+17,2432+17):
        Y.append((float(stock_data.loc[i, ['調整後終値']])*100 / float(stock_data.loc[i-1, ['調整後終値']]))-100)
    for i in range(30,2432):
        if Y[i-30]>0:
            answers.append(1)
        else:
            answers.append(0)
    #Y = pd.read_csv('modified_answer_9613.csv', encoding="shift-jis")
    print(Y)
    y_train.append(answers[0:2175])
    if kobetu == check_meigara:
        y_test.append(answers[2175:2402])
y_train = np_utils.to_categorical(y_train, 2)
y_train = y_train.reshape(-1, 2)
y_train = tf.convert_to_tensor(y_train, np.float32)
y_test = np_utils.to_categorical(y_test, 2)
y_test = y_test.reshape(-1, 2)
y_test = tf.convert_to_tensor(y_test, np.float32)
print(y_train)
print(y_test)



#モデルの構造を定義
model = Sequential()
model.add(Dense(512, input_shape=(3072,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(2))
model.add(Activation('softmax'))



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


img_pred = model.predict(X_test, steps=5)
print(img_pred)
