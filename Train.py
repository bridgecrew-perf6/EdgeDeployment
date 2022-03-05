# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from Model_Builder import Models
import TF_TRT_Converter 
import os

#import numpy as np

print("packages imported")

(X_train, y_train), (X_test, y_test) = mnist.load_data()
"""
# visualize some of the data
fig,ax = plt.subplots(1,5)
for i in range(5):
    ax[i].imshow(X_train[i],cmap='gray')
"""
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

with open("data.p", "wb") as f:
    pickle.dump(data, f)


class model_training():
    
    def __init__(self, data):
        with open("model_dict.p", "rb") as f:
            self.model_dict = pickle.load(f)
        self.input_shape = 784
        self.data = data
    
    """
    def train_conv_model(self, model, model_name):
        X_train, y_train, X_test, y_test = self.data
        X_train.reshape(28, 28, 1)
        X_test.reshape(28, 28, 1)
        model.compile(
            loss = tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer = "adam",
            metrics = ['accuracy']
            )
        with open("model_summaries/" + model_name + ".txt", "w") as f:
            with redirect_stdout(f):
                model.summary()
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        mc = ModelCheckpoint("models/" + model_name + '_best_model.h5', 
                             monitor='val_loss', 
                             mode='min', 
                             verbose=1
                             )
        print("training model...")
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=128,
            epochs=500,
            callbacks=[es, mc],
            validation_data=(X_test, y_test)
        )
        #tf.keras.models.save_model(model, "pb_models/" + model_name + ".pb")
        tf.saved_model.save(model, "pb_models/" + model_name + ".pb")
        return history
    """

    def train_c1nn_model(self, model, model_name):
        X_train, y_train, X_test, y_test = self.data
        #X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        #X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        model.compile(
            loss = tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer = "adam",
            metrics = ['accuracy']
            )
        with open("model_summaries/" + model_name + ".txt", "w") as f:
            with redirect_stdout(f):
                model.summary()
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        mc = ModelCheckpoint("models/" + model_name + '_best_model.h5', 
                             monitor='val_loss', 
                             mode='min', 
                             verbose=1
                             )
        print("training model...")
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=128,
            epochs=500,
            callbacks=[es, mc],
            validation_data=(X_test, y_test)
        )
        #tf.keras.models.save_model(model, "pb_models/" + model_name + ".pb")
        tf.saved_model.save(model, "pb_models/c1nn/" + model_name + ".pb")
        return history
    

    def train_dnn_model(self, model, model_name):
        X_train, y_train, X_test, y_test = self.data
        model.compile(
            loss = tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer = "adam",
            metrics = ['accuracy']
            )
        with open("model_summaries/" + model_name + ".txt", "w") as f:
            with redirect_stdout(f):
                model.summary()
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        mc = ModelCheckpoint("models/" + model_name + '_best_model.h5', 
                             monitor='val_loss', 
                             mode='min', 
                             verbose=1
                             )
        print("training model...")
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=128,
            epochs=500,
            callbacks=[es, mc],
            validation_data=(X_test, y_test)
        )
        #tf.keras.models.save_model(model, "pb_models/" + model_name + ".pb")
        tf.saved_model.save(model, "pb_models/dnn/" + model_name + ".pb)
        return history
    
    def training_plots(self, history, model_name):
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy: ' + model_name)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig("plots/" + model_name + "_accuracy.jpg")
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss: ' + model_name)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig("plots/" + model_name + "_loss.jpg")

    
    def save_history(self, history, model_name):
        with open(model_name + ".p", "wb") as f:
            pickle.dump(history, f)


    def run(self):
        
        model = Models.build_dnn_128_3L(self.input_shape)
        D128_3 = self.train_dnn_model(model, "dnn_128_3L")
        self.training_plots(D128_3, "3 Layer Small")
        
        model = Models.build_dnn_512_3L(self.input_shape)
        D512_3 = self.train_dnn_model(model, "dnn_512_3L")
        self.training_plots(D512_3, "3 Layer Medium")
        model = Models.build_dnn_1024_3L(self.input_shape)
        D1024_3 = self.train_dnn_model(model, "dnn_1024_3L")
        self.training_plots(D1024_3, "3 Layer Large")
        
        model = Models.build_dnn_128_6L(self.input_shape)
        D128_6 = self.train_dnn_model(model, "dnn_128_6L")
        self.training_plots(D128_6, "6 Layer Small")
        model = Models.build_dnn_512_6L(self.input_shape)
        D512_6 = self.train_dnn_model(model, "dnn_512_6L")
        self.training_plots(D512_6, "6 Layer Medium")
        model = Models.build_dnn_1024_6L(self.input_shape)
        D1024_6 = self.train_dnn_model(model, "dnn_1024_6L")
        self.training_plots(D1024_6, "6 Layer Large")
        
        model = Models.build_dnn_1024_3L(self.input_shape)
        D128_12 = self.train_dnn_model(model, "dnn_128_12L")
        #self.save_history(D1024_6, "dnn_128_12L")
        self.training_plots(D128_12, "12 Layer Small")
        model = Models.build_dnn_1024_6L(self.input_shape)
        D512_12 = self.train_dnn_model(model, "dnn_512_12L")
        #self.save_history(D1024_6, "dnn_512_12L")
        self.training_plots(D512_12, "12 Layer Medium")
        model = Models.build_dnn_1024_12L(self.input_shape)
        D1024_12 = self.train_dnn_model(model, "dnn_1024_12L")
        #self.save_history(D1024_6, "dnn_1024_12L")
        self.training_plots(D1024_12, "12 Layer Large")
        
        model = Models.build_c1nn32(self.input_shape)
        C32 = self.train_c1nn_model(model, "c1nn32")
        #self.save_history(D1024_6, "c1nn32")
        self.training_plots(C32, "C1NN_32")
        model = Models.build_c1nn32_16(self.input_shape)
        C32_16 = self.train_c1nn_model(model, "c1nn32_16")
        #self.save_history(D1024_6, "c1nn32/16")
        self.training_plots(C32_16, "C1NN_32_16")
        
        model = Models.build_c1nn64(self.input_shape)
        C64 = self.train_c1nn_model(model, "c1nn64")
        #self.save_history(D1024_6, "c1nn64")
        self.training_plots(C64, "C1NN_64")
        model = Models.build_c1nn64_32(self.input_shape)
        C64_32 = self.train_c1nn_model(model, "c1nn64_32")
        #self.save_history(D1024_6, "c1nn64/32")
        self.training_plots(C64_32, "C1NN_64_32")
        model = Models.build_c1nn64_32_16(self.input_shape)
        C64_32_16 = self.train_c1nn_model(model, "c1nn64_32_16")
        #self.save_history(D1024_6, "c1nn64/32/16")
        self.training_plots(C64_32_16, "C1NN_64_32_16")
        
        model = Models.build_c1nn128(self.input_shape)
        C128 = self.train_c1nn_model(model, "c1nn128")
        #self.save_history(D1024_6, "c1nn128")
        self.training_plots(C128, "C1NN_128")
        model = Models.build_c1nn128_64(self.input_shape)
        C128_64 = self.train_c1nn_model(model, "c1nn128_64")
        #self.save_history(D1024_6, "c1nn128/64")
        self.training_plots(C128_64, "C1NN_128_64")  
        model = Models.build_c1nn128_64_32(self.input_shape)
        C128_64_32 = self.train_c1nn_model(model, "c1nn128_64_32")
        #self.save_history(D1024_6, "c1nn6128/64/32")
        self.training_plots(C128_64_32, "C1NN_128_64_32")  
        model = Models.build_c1nn128_64_32_16(self.input_shape)
        C128_64_32_16 = self.train_c1nn_model(model, "c1nn128_64_32_16")
        #self.save_history(D1024_6, "c1nn128/64/32/16")
        self.training_plots(C128_64_32_16, "C1NN_128_64_32_16") 
        
        """
        model = Models.build_c1nn128_64_32_16(self.input_shape)
        CONV32 = self.train_conv_model(model, "conv32")
        #self.save_history(CONV32, "conv32")
        self.training_plots(CONV32, "conv32") 
        """
        
        return (D128_3, D512_3, D1024_3, D128_6, D512_6, D1024_6, D128_12, 
                D512_12, D1024_12, C32, C32_16, C64, C64_32, C64_32_16, C128,  
                C128_64, C128_64_32, C128_64_32_16
                )
        
        
        

        
        
if __name__ == "__main__":
    MT = model_training(data)
    models = MT.run()
    #test = tf.keras.models.load_model("models/trt_test.pb")
    #preds = test.predict(X_train)
    
    

    
""" 
#TODO: Already moved to training functions, but also need way to get params of converted models
from contextlib import redirect_stdout
to_convert = os.listdir("trt_models/dnn/")
for m in to_convert:
    temp_model = tf.keras.models.load_model("/home/msweber/Code/EdgeDeployment/trt_models/dnn/" + m)
    with open("/home/msweber/Code/EdgeDeployment/model_summaries/trt_models/" + m[:-3] + ".txt", "w") as f:
        with redirect_stdout(f):
            temp_model.summary()
to_convert = os.listdir("trt_models/c1nn/")
for m in to_convert:
    temp_model = tf.keras.models.load_model("/home/msweber/Code/EdgeDeployment/trt_models/c1nn/" + m)
    with open("/home/msweber/Code/EdgeDeployment/model_summaries/trt_models/" + m[:-3] + ".txt", "w") as f:
        with redirect_stdout(f):
            temp_model.summary()
"""