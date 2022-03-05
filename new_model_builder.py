#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 10:31:50 2022

@author: msweber
"""

import tensorflow as tf

class Models():
    
    def __init__(self):
        pass
    
    def build_dnn_128_3L(self):
        print("building 128/64 DNN model...")
        inputs = tf.keras.Input(shape=(28,28,1), name='input')
        # [28, 28, 1] => [28, 28, 64]
        input = tf.keras.layers.Flatten(name="flatten")(inputs)
        fc_1 = tf.keras.layers.Dense(128, activation='relu', name='fc_1')(input)
        fc_2 = tf.keras.layers.Dense(64, activation='relu', name='fc_2')(fc_1)
        pred = tf.keras.layers.Dense(10, activation='softmax', name='output')(fc_2)

        model = tf.keras.Model(inputs=inputs, outputs=pred, name='mnist')
        return model
    
    def build_dnn_512_3L(self):
        print("building 512/256 DNN model...")
        inputs = tf.keras.Input(shape=(28,28,1), name='input')
        # [28, 28, 1] => [28, 28, 64]
        input = tf.keras.layers.Flatten(name="flatten")(inputs)
        fc_1 = tf.keras.layers.Dense(512, activation='relu', name='fc_1')(input)
        fc_2 = tf.keras.layers.Dense(256, activation='relu', name='fc_2')(fc_1)
        pred = tf.keras.layers.Dense(10, activation='softmax', name='output')(fc_2)

        model = tf.keras.Model(inputs=inputs, outputs=pred, name='mnist')
        return model

        
    def build_dnn_1024_3L(self):
        print("building 1024/512 DNN model...")
        inputs = tf.keras.Input(shape=(28,28,1), name='input')
        # [28, 28, 1] => [28, 28, 64]
        input = tf.keras.layers.Flatten(name="flatten")(inputs)
        fc_1 = tf.keras.layers.Dense(1024, activation='relu', name='fc_1')(input)
        fc_2 = tf.keras.layers.Dense(512, activation='relu', name='fc_2')(fc_1)
        pred = tf.keras.layers.Dense(10, activation='softmax', name='output')(fc_2)

        model = tf.keras.Model(inputs=inputs, outputs=pred, name='mnist')
        return model
     

    def build_dnn_128_6L(self):
        print("building 128/96/64/32/16 DNN model...")
        inputs = tf.keras.Input(shape=(28,28,1), name='input')
        # [28, 28, 1] => [28, 28, 64]
        input = tf.keras.layers.Flatten(name="flatten")(inputs)
        fc_1 = tf.keras.layers.Dense(128, activation='relu', name='fc_1')(input)
        fc_2 = tf.keras.layers.Dense(96, activation='relu', name='fc_2')(fc_1)
        fc_3 = tf.keras.layers.Dense(64, activation='relu', name='fc_3')(fc_2)
        fc_4 = tf.keras.layers.Dense(32, activation='relu', name='fc_4')(fc_3)
        fc_5 = tf.keras.layers.Dense(16, activation='relu', name='fc_5')(fc_4)
        pred = tf.keras.layers.Dense(10, activation='softmax', name='output')(fc_5)

        model = tf.keras.Model(inputs=inputs, outputs=pred, name='mnist')
        return model
    
     
    def build_dnn_512_6L(self):
        print("building 512/256/128/64/32 DNN model...")
        inputs = tf.keras.Input(shape=(28,28,1), name='input')
        # [28, 28, 1] => [28, 28, 64]
        input = tf.keras.layers.Flatten(name="flatten")(inputs)
        fc_1 = tf.keras.layers.Dense(512, activation='relu', name='fc_1')(input)
        fc_2 = tf.keras.layers.Dense(256, activation='relu', name='fc_2')(fc_1)
        fc_3 = tf.keras.layers.Dense(128, activation='relu', name='fc_3')(fc_2)
        fc_4 = tf.keras.layers.Dense(64, activation='relu', name='fc_4')(fc_3)
        fc_5 = tf.keras.layers.Dense(32, activation='relu', name='fc_5')(fc_4)
        pred = tf.keras.layers.Dense(10, activation='softmax', name='output')(fc_5)

        model = tf.keras.Model(inputs=inputs, outputs=pred, name='mnist')
        return model
    
     
    def build_dnn_1024_6L(self):
        print("building 1024/512/256/128/64 DNN model...")
        inputs = tf.keras.Input(shape=(28,28,1), name='input')
        # [28, 28, 1] => [28, 28, 64]
        input = tf.keras.layers.Flatten(name="flatten")(inputs)
        fc_1 = tf.keras.layers.Dense(1024, activation='relu', name='fc_1')(input)
        fc_2 = tf.keras.layers.Dense(512, activation='relu', name='fc_2')(fc_1)
        fc_3 = tf.keras.layers.Dense(128, activation='relu', name='fc_3')(fc_2)
        fc_4 = tf.keras.layers.Dense(64, activation='relu', name='fc_4')(fc_3)
        fc_5 = tf.keras.layers.Dense(32, activation='relu', name='fc_5')(fc_4)
        pred = tf.keras.layers.Dense(10, activation='softmax', name='output')(fc_5)

        model = tf.keras.Model(inputs=inputs, outputs=pred, name='mnist')
        return model
    
    
    def build_dnn_128_12L(self):
        print("building 128/128/96/96/64/64/48/48/32/32/16 DNN model...")
        inputs = tf.keras.Input(shape=(28,28,1), name='input')
        # [28, 28, 1] => [28, 28, 64]
        input = tf.keras.layers.Flatten(name="flatten")(inputs)
        fc_1 = tf.keras.layers.Dense(128, activation='relu', name='fc_1')(input)
        fc_2 = tf.keras.layers.Dense(128, activation='relu', name='fc_2')(fc_1)
        fc_3 = tf.keras.layers.Dense(96, activation='relu', name='fc_3')(fc_2)
        fc_4 = tf.keras.layers.Dense(96, activation='relu', name='fc_4')(fc_3)
        fc_5 = tf.keras.layers.Dense(64, activation='relu', name='fc_5')(fc_4)
        fc_6 = tf.keras.layers.Dense(64, activation='relu', name='fc_6')(fc_5)
        fc_7 = tf.keras.layers.Dense(48, activation='relu', name='fc_7')(fc_6)
        fc_8 = tf.keras.layers.Dense(48, activation='relu', name='fc_8')(fc_7)
        fc_9 = tf.keras.layers.Dense(32, activation='relu', name='fc_9')(fc_8)
        fc_10 = tf.keras.layers.Dense(32, activation='relu', name='fc_10')(fc_9)
        fc_11 = tf.keras.layers.Dense(16, activation='relu', name='fc_11')(fc_10)
        pred = tf.keras.layers.Dense(10, activation='softmax', name='output')(fc_11)

        model = tf.keras.Model(inputs=inputs, outputs=pred, name='mnist')
        return model
    
     
    def build_dnn_512_12L(self):
        print("building 512/512/256/256/128/128/64/64/32/32/16 DNN model...")
        inputs = tf.keras.Input(shape=(28,28,1), name='input')
        # [28, 28, 1] => [28, 28, 64]
        input = tf.keras.layers.Flatten(name="flatten")(inputs)
        fc_1 = tf.keras.layers.Dense(512, activation='relu', name='fc_1')(input)
        fc_2 = tf.keras.layers.Dense(512, activation='relu', name='fc_2')(fc_1)
        fc_3 = tf.keras.layers.Dense(256, activation='relu', name='fc_3')(fc_2)
        fc_4 = tf.keras.layers.Dense(256, activation='relu', name='fc_4')(fc_3)
        fc_5 = tf.keras.layers.Dense(128, activation='relu', name='fc_5')(fc_4)
        fc_6 = tf.keras.layers.Dense(128, activation='relu', name='fc_6')(fc_5)
        fc_7 = tf.keras.layers.Dense(64, activation='relu', name='fc_7')(fc_6)
        fc_8 = tf.keras.layers.Dense(64, activation='relu', name='fc_8')(fc_7)
        fc_9 = tf.keras.layers.Dense(32, activation='relu', name='fc_9')(fc_8)
        fc_10 = tf.keras.layers.Dense(32, activation='relu', name='fc_10')(fc_9)
        fc_11 = tf.keras.layers.Dense(16, activation='relu', name='fc_11')(fc_10)
        pred = tf.keras.layers.Dense(10, activation='softmax', name='output')(fc_11)

        model = tf.keras.Model(inputs=inputs, outputs=pred, name='mnist')
        return model
     
        
    def build_dnn_1024_12L(self):
        print("building 1024/1024/512/512/256/256/128/128/64/32/16 DNN model...")
        inputs = tf.keras.Input(shape=(28,28,1), name='input')
        # [28, 28, 1] => [28, 28, 64]
        input = tf.keras.layers.Flatten(name="flatten")(inputs)
        fc_1 = tf.keras.layers.Dense(1024, activation='relu', name='fc_1')(input)
        fc_2 = tf.keras.layers.Dense(1024, activation='relu', name='fc_2')(fc_1)
        fc_3 = tf.keras.layers.Dense(512, activation='relu', name='fc_3')(fc_2)
        fc_4 = tf.keras.layers.Dense(512, activation='relu', name='fc_4')(fc_3)
        fc_5 = tf.keras.layers.Dense(256, activation='relu', name='fc_5')(fc_4)
        fc_6 = tf.keras.layers.Dense(256, activation='relu', name='fc_6')(fc_5)
        fc_7 = tf.keras.layers.Dense(128, activation='relu', name='fc_7')(fc_6)
        fc_8 = tf.keras.layers.Dense(128, activation='relu', name='fc_8')(fc_7)
        fc_9 = tf.keras.layers.Dense(64, activation='relu', name='fc_9')(fc_8)
        fc_10 = tf.keras.layers.Dense(32, activation='relu', name='fc_10')(fc_9)
        fc_11 = tf.keras.layers.Dense(16, activation='relu', name='fc_11')(fc_10)
        pred = tf.keras.layers.Dense(10, activation='softmax', name='output')(fc_11)

        model = tf.keras.Model(inputs=inputs, outputs=pred, name='mnist')
        return model
     
     
    def build_c1nn32(self):
        print("building 32 1DCNN model...")
        inputs = tf.keras.Input(shape=(28,28,1), name='input')
        # [28, 28, 1] => [28, 28, 64]
        input = tf.keras.layers.Flatten(name="flatten")(inputs)
        input = tf.expand_dims(input, axis=2)
        fc_1 = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_1')(input)
        fc_2 = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_2')(fc_1)
        fc_3 = tf.keras.layers.Dropout(0.2, name='fc_3')(fc_2)
        fc_4 = tf.keras.layers.MaxPooling1D(pool_size=2, name='fc_4')(fc_3)
        fc_5 = tf.keras.layers.Flatten(name='fc5')(fc_4)
        pred = tf.keras.layers.Dense(10, activation='softmax', name='output')(fc_5)

        model = tf.keras.Model(inputs=inputs, outputs=pred, name='mnist')
        return model
    
    
    def build_c1nn32_16(self):
        print("building 32/16 1DCNN model...")
        inputs = tf.keras.Input(shape=(28,28,1), name='input')
        # [28, 28, 1] => [28, 28, 64]
        input = tf.keras.layers.Flatten(name="flatten")(inputs)
        input = tf.expand_dims(input, axis=2)
        fc_1 = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_1')(input)
        fc_2 = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_2')(fc_1)
        fc_3 = tf.keras.layers.Dropout(0.2, name='fc_3')(fc_2)
        fc_4 = tf.keras.layers.MaxPooling1D(pool_size=2, name='fc_4')(fc_3)
        fc_5 = tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_5')(fc_4)
        fc_6 = tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_6')(fc_5)
        fc_7 = tf.keras.layers.Dropout(0.2, name='fc_7')(fc_6)
        fc_8 = tf.keras.layers.MaxPooling1D(pool_size=2, name='fc_8')(fc_7)
        fc_9 = tf.keras.layers.Flatten(name='fc9')(fc_8)
        pred = tf.keras.layers.Dense(10, activation='softmax', name='output')(fc_9)

        model = tf.keras.Model(inputs=inputs, outputs=pred, name='mnist')
        return model


    def build_c1nn64(self):
        print("building 64 1DCNN model...")
        inputs = tf.keras.Input(shape=(28,28,1), name='input')
        # [28, 28, 1] => [28, 28, 64]
        input = tf.keras.layers.Flatten(name="flatten")(inputs)
        input = tf.expand_dims(input, axis=2)
        fc_1 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_1')(input)
        fc_2 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_2')(fc_1)
        fc_3 = tf.keras.layers.Dropout(0.2, name='fc_3')(fc_2)
        fc_4 = tf.keras.layers.MaxPooling1D(pool_size=2, name='fc_4')(fc_3)
        fc_5 = tf.keras.layers.Flatten(name='fc5')(fc_4)
        pred = tf.keras.layers.Dense(10, activation='softmax', name='output')(fc_5)

        model = tf.keras.Model(inputs=inputs, outputs=pred, name='mnist')
        return model


    def build_c1nn64_32(self):
        print("building 64/32 1DCNN model...")
        inputs = tf.keras.Input(shape=(28,28,1), name='input')
        # [28, 28, 1] => [28, 28, 64]
        input = tf.keras.layers.Flatten(name="flatten")(inputs)
        input = tf.expand_dims(input, axis=2)
        fc_1 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_1')(input)
        fc_2 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_2')(fc_1)
        fc_3 = tf.keras.layers.Dropout(0.2, name='fc_3')(fc_2)
        fc_4 = tf.keras.layers.MaxPooling1D(pool_size=2, name='fc_4')(fc_3)
        fc_5 = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_5')(fc_4)
        fc_6 = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_6')(fc_5)
        fc_7 = tf.keras.layers.Dropout(0.2, name='fc_7')(fc_6)
        fc_8 = tf.keras.layers.MaxPooling1D(pool_size=2, name='fc_8')(fc_7)
        fc_9 = tf.keras.layers.Flatten(name='fc9')(fc_8)
        pred = tf.keras.layers.Dense(10, activation='softmax', name='output')(fc_9)

        model = tf.keras.Model(inputs=inputs, outputs=pred, name='mnist')
        return model
    
    
    def build_c1nn64_32_16(self):
        print("building 64/32/16 1DCNN model...")
        inputs = tf.keras.Input(shape=(28,28,1), name='input')
        # [28, 28, 1] => [28, 28, 64]
        input = tf.keras.layers.Flatten(name="flatten")(inputs)
        input = tf.expand_dims(input, axis=2)
        fc_1 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_1')(input)
        fc_2 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_2')(fc_1)
        fc_3 = tf.keras.layers.Dropout(0.2, name='fc_3')(fc_2)
        fc_4 = tf.keras.layers.MaxPooling1D(pool_size=2, name='fc_4')(fc_3)
        fc_5 = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_5')(fc_4)
        fc_6 = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_6')(fc_5)
        fc_7 = tf.keras.layers.Dropout(0.2, name='fc_7')(fc_6)
        fc_8 = tf.keras.layers.MaxPooling1D(pool_size=2, name='fc_8')(fc_7)
        fc_9 = tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_9')(fc_8)
        fc_10 = tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_10')(fc_9)
        fc_11 = tf.keras.layers.Dropout(0.2, name='fc_11')(fc_10)
        fc_12 = tf.keras.layers.MaxPooling1D(pool_size=2, name='fc_12')(fc_11)
        fc_13 = tf.keras.layers.Flatten(name='fc13')(fc_12)
        pred = tf.keras.layers.Dense(10, activation='softmax', name='output')(fc_13)

        model = tf.keras.Model(inputs=inputs, outputs=pred, name='mnist')
        return model


    def build_c1nn128_64_32(self):
        print("building 128/64/32 1DCNN model...")
        inputs = tf.keras.Input(shape=(28,28,1), name='input')
        # [28, 28, 1] => [28, 28, 64]
        input = tf.keras.layers.Flatten(name="flatten")(inputs)
        input = tf.expand_dims(input, axis=2)
        fc_1 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_1')(input)
        fc_2 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_2')(fc_1)
        fc_3 = tf.keras.layers.Dropout(0.2, name='fc_3')(fc_2)
        fc_4 = tf.keras.layers.MaxPooling1D(pool_size=2, name='fc_4')(fc_3)
        fc_5 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_5')(fc_4)
        fc_6 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_6')(fc_5)
        fc_7 = tf.keras.layers.Dropout(0.2, name='fc_7')(fc_6)
        fc_8 = tf.keras.layers.MaxPooling1D(pool_size=2, name='fc_8')(fc_7)
        fc_9 = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_9')(fc_8)
        fc_10 = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', 
                                      kernel_initializer='normal', 
                                      kernel_regularizer="l2", 
                                      name='fc_10')(fc_9)
        fc_11 = tf.keras.layers.Dropout(0.2, name='fc_11')(fc_10)
        fc_12 = tf.keras.layers.MaxPooling1D(pool_size=2, name='fc_12')(fc_11)
        fc_13 = tf.keras.layers.Flatten(name='fc13')(fc_12)
        pred = tf.keras.layers.Dense(10, activation='softmax', name='output')(fc_13)

        model = tf.keras.Model(inputs=inputs, outputs=pred, name='mnist')
        return model