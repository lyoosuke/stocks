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


X_train = image_reshape[0:2001]
X_test = image_reshape[2001:2403]
X_train /= 255
X_test /= 255



Y = pd.read_csv('modified_answer_9613.csv', encoding="shift-jis")
print(Y)
y_train = Y[0:2001]
y_test = Y[2001:2403]
y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)
print(y_train)
print(y_test)



#モデルの構造を定義
model = Sequential()
model.add(Dense(512, input_shape=(307200,)))
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




