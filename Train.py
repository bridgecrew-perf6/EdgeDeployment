# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
#import numpy as np

print("packages imported")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# visualize some of the data
fig,ax = plt.subplots(1,5)
for i in range(5):
    ax[i].imshow(X_train[i],cmap='gray')

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# normalize and flatten images
X_train = X_train.reshape(-1,28*28)/ 255.0
X_test = X_test.reshape(-1,28*28)/ 255.0
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
data = X_train, y_train, X_test, y_test


class model_training():
    
    def __init__(self, data):
        with open("model_dict.p", "rb") as f:
            self.model_dict = pickle.load(f)
        self.input_shape = 784
        self.data = data

    
    def build_model(self, arch):
        
        # functional API
        print("building model...")
        input_layer = tf.keras.Input(shape=(self.input_shape))
        for i in range(len(arch["Layer_Types"])):
            x = input_layer
            if arch["Layer_Types"][i] == "Dense":
                x = tf.keras.layers.arch["Layer_Types"][i](filters=arch["Layer_Nodes"][i], \
                    activation=arch["Layer_Activations"][i])(x)
            elif ach["Layer_Types"][i] == "Conv1D":
                x = tf.keras.layers.arch["Layer_Types"][i](arch["Layer_Nodes"][i][1], \
                    kernel_size=arch["Layer_Nodes"][i][2], \
                    activation=arch["Layer_Activations"][i])(x)
            elif arch["Layer_Types"][i] == "MaxPooling1D":
                x = tf.keras.layers.MaxPooling1D(pool_size=arch["Layer_Nodes"][i])(x)
            elif arch["Layer_Types"][i] == "Flatten":
                x = tf.keras.layers.Flatten()(x)
            elif arch["Layer_Types"][i] == "Dropout":
                x = tf.keras.layers.Dropout(arch["Layer_Nodes"][i])
            elif arch["Layer_Types"][i] == "Output":
                output = tf.keras.layers.Dense(10,activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        model.compile(
            loss = tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer = arch["Optimization"],
            metrics = ['accuracy']
            )
        return model
    
    def train_model(self, arch, model):
        X_train, y_train, X_test, y_test = self.data
        print("training model...")
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=128,
            epochs=500,
            validation_data=(X_test, y_test)
        )
        return history
    
    def training_plots(self, history):
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    
    def save_models(self):
        with open("model_dict2.p", "wb") as f:
            pickle.dump(self.model_dict, f)

    def run(self):
        count = 0
        while count <3:
            print("starting build...")
            for model_no, value in self.model_dict.items():
                print(model_no)
                arch = value['Arch']
                model = self.build_model(arch)
                history = self.train_model(arch, model)
                self.training_plots(history)
                self.save_models()
                count += 1
        
        
if __name__ == "__main__":
    MT = model_training(data)
    with open("model_dict.p", "rb") as f:
        model_dict = pickle.load(f)
    MT.run()
