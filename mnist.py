import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# visualize some of the data
fig,ax = plt.subplots(1,5)
for i in range(5):
    ax[i].imshow(X_train[i],cmap='gray')

print(X_train.shape)
print(X_test.shape)

# normalize and flatten images
X_train = X_train.reshape(-1,28*28)/ 255.0
X_test = X_test.reshape(-1,28*28)/ 255.0
print(X_train.shape)
print(X_test.shape)

# functional API
inputs = tf.keras.Input(shape=(784))
x = tf.keras.layers.Dense(512, activation='relu')(inputs)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
output = tf.keras.layers.Dense(10,activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=output)

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = 'adam',
    metrics = ['accuracy']
)

model.fit(
    x=X_train,
    y=y_train,
    batch_size=128,
    epochs=500,
    validation_data=(X_test, y_test)
)

print(model.summary())

loss = pd.DataFrame(model.history.history)
loss.plot()

model.evaluate(X_test, y_test, batch_size=128,verbose=2)

