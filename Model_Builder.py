#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 10:31:50 2022

@author: msweber
"""

import tensorflow as tf

class Models():
    
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_dnn_128_3L(self):
        print("building 128/64 DNN model...")
        inputs = tf.keras.Input(shape=784)
        x = tf.keras.layers.Dense(128, activation='relu', 
                                  kernel_initializer='normal', 
                                  kernel_regularizer="l2"
                                  )(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(64, activation='relu', 
                                  kernel_initializer='normal', 
                                  kernel_regularizer="l2"
                                  )(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output = tf.keras.layers.Dense(10,activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model
    
    def build_dnn_512_3L(self):
         print("building 512/256 DNN model...")
         inputs = tf.keras.Input(shape=784)
         x = tf.keras.layers.Dense(512, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(inputs)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(256, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model

        
    def build_dnn_1024_3L(self):
         print("building 1024/512 DNN model...")
         inputs = tf.keras.Input(shape=784)
         x = tf.keras.layers.Dense(1024, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(inputs)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(512, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model
     

    def build_dnn_128_6L(self):
         print("building 128/96/64/32/16 DNN model...")
         inputs = tf.keras.Input(shape=784)
         x = tf.keras.layers.Dense(128, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(inputs)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(96, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(64, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(32, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(16, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model
    
     
    def build_dnn_512_6L(self):
         print("building 512/256/128/64/32 DNN model...")
         inputs = tf.keras.Input(shape=784)
         x = tf.keras.layers.Dense(512, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(inputs)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(256, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(128, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(64, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(32, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model
    
     
    def build_dnn_1024_6L(self):
         print("building 1024/512/256/128/64 DNN model...")
         inputs = tf.keras.Input(shape=784)
         x = tf.keras.layers.Dense(1024, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(inputs)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(512, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(256, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(128, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(64, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model
    
    
    def build_dnn_128_12L(self):
         print("building 128/128/96/96/64/64/32/32/16/16/16 DNN model...")
         inputs = tf.keras.Input(shape=784)
         x = tf.keras.layers.Dense(128, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(inputs)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(128, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(96, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(96, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(64, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(64, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(32, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(32, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(16, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(16, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(16, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model    
     
     
    def build_dnn_512_12L(self):
         print("building 512/512/256/256/128/128/64/64/32/32/16 DNN model...")
         inputs = tf.keras.Input(shape=784)
         x = tf.keras.layers.Dense(512, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(inputs)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(512, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(256, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(256, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(128, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(128, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(64, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(64, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(32, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(32, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(16, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model
     
        
    def build_dnn_1024_12L(self):
         print("building 1024/1024/512/512/256/256/128/128/64/64/32 DNN model...")
         inputs = tf.keras.Input(shape=784)
         x = tf.keras.layers.Dense(1024, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(inputs)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(1024, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(512, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(512, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(256, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(256, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(128, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(128, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(64, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(64, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         x = tf.keras.layers.Dense(32, activation='relu', 
                                   kernel_initializer='normal', 
                                   kernel_regularizer="l2"
                                   )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.BatchNormalization()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model 
     
     
    def build_c1nn32(self):
         print("building 32 1DCNN model...")
         inputs = tf.keras.Input(shape=(784,1))
         x = tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(inputs)
         x = tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.4)(x)
         x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
         x = tf.keras.layers.Flatten()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model
     
    
    def build_c1nn32_16(self):
         print("building 32/16 1DCNN model...")
         inputs = tf.keras.Input(shape=(784,1))
         x = tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(inputs)
         x = tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
         x = tf.keras.layers.Conv1D(filters=16,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(inputs)
         x = tf.keras.layers.Conv1D(filters=16,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(inputs)
         x = tf.keras.layers.Dropout(0.4)(x)
         x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
         x = tf.keras.layers.Flatten()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model
     
     
    def build_c1nn64(self):
         print("building 64 1DCNN model...")
         inputs = tf.keras.Input(shape=(784,1))
         x = tf.keras.layers.Conv1D(filters=64,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(inputs)
         x = tf.keras.layers.Conv1D(filters=64,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.4)(x)
         x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
         x = tf.keras.layers.Flatten()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model
     
     
    def build_c1nn64_32(self):
         print("building 64/32 1DCNN model...")
         inputs = tf.keras.Input(shape=(784,1))
         x = tf.keras.layers.Conv1D(filters=64,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(inputs)
         x = tf.keras.layers.Conv1D(filters=64,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.4)(x)
         x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
         x = tf.keras.layers.Flatten()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model
     
     
    def build_c1nn64_32_16(self):
         print("building 64/32/16 1DCNN model...")
         inputs = tf.keras.Input(shape=(784,1))
         x = tf.keras.layers.Conv1D(filters=64,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(inputs)
         x = tf.keras.layers.Conv1D(filters=64,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.Conv1D(filters=16,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Conv1D(filters=16,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.4)(x)
         x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
         x = tf.keras.layers.Flatten()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model
    
    
    def build_c1nn128(self):
         print("building 128 1DCNN model...")
         inputs = tf.keras.Input(shape=(784,1))
         x = tf.keras.layers.Conv1D(filters=128,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(inputs)
         x = tf.keras.layers.Conv1D(filters=128,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.4)(x)
         x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
         x = tf.keras.layers.Flatten()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model
     
     
    def build_c1nn128_64(self):
         print("building 128/64 1DCNN model...")
         inputs = tf.keras.Input(shape=(784,1))
         x = tf.keras.layers.Conv1D(filters=128,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(inputs)
         x = tf.keras.layers.Conv1D(filters=128,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.Conv1D(filters=64,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Conv1D(filters=64,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.4)(x)
         x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
         x = tf.keras.layers.Flatten()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model
     
     
    def build_c1nn128_64_32(self):
         print("building 128/64/32 1DCNN model...")
         inputs = tf.keras.Input(shape=(784,1))
         x = tf.keras.layers.Conv1D(filters=128,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(inputs)
         x = tf.keras.layers.Conv1D(filters=128,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.Conv1D(filters=64,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Conv1D(filters=64,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.4)(x)
         x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
         x = tf.keras.layers.Flatten()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model
     
     
    def build_c1nn128_64_32_16(self):
         print("building 128/64/32/16 1DCNN model...")
         inputs = tf.keras.Input(shape=(784,1))
         x = tf.keras.layers.Conv1D(filters=128,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(inputs)
         x = tf.keras.layers.Conv1D(filters=128,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.Conv1D(filters=64,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Conv1D(filters=64,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.Conv1D(filters=16,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Conv1D(filters=16,
                                    kernel_size=3,
                                    activation='relu', 
                                    kernel_initializer='normal', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.4)(x)
         x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
         x = tf.keras.layers.Flatten()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model
     

    def build_cnn32(self):
         print("building 32/16 1DCNN model...")
         inputs = tf.keras.Input(shape=(28, 28 ,1))
         x = tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=(3, 3),
                                    activation='relu', 
                                    kernel_initializer='he_uniform', 
                                    kernel_regularizer="l2"
                                    )(inputs)
         x = tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=(3, 3),
                                    activation='relu', 
                                    kernel_initializer='he_uniform', 
                                    kernel_regularizer="l2"
                                    )(x)
         x = tf.keras.layers.Dropout(0.2)(x)
         x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
         x = tf.keras.layers.Conv1D(filters=16,
                                    kernel_size=(3, 3),
                                    activation='relu', 
                                    kernel_initializer='he_uniform', 
                                    kernel_regularizer="l2"
                                    )(inputs)
         x = tf.keras.layers.Conv1D(filters=16,
                                    kernel_size=(3, 3),
                                    activation='relu', 
                                    kernel_initializer='he_uniform', 
                                    kernel_regularizer="l2"
                                    )(inputs)
         x = tf.keras.layers.Dropout(0.4)(x)
         x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
         x = tf.keras.layers.Flatten()(x)
         output = tf.keras.layers.Dense(10,activation='softmax')(x)
         model = tf.keras.Model(inputs=inputs, outputs=output)
         return model