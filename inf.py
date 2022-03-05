#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:11:41 2022

@author: msweber
"""

import tensorflow as tf
import pickle
import argparse
import os
import numpy as np
import time


# Set Working Directory
wd = "/home/msweber/Code/"
"""
args = None
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True,
	help="base directory, which contains EdgeDeployment")
#p.add_argument("-p", "--path", required=True,
#	help="path to location of model files and output")
args = vars(ap.parse_args())

wd = args["directory"]  
"""
os.chdir(wd)


# Define timed prediction function
def make_preds(model, X):
    tic = time.perf_counter()
    preds = model.predict(X)
    toc = time.perf_counter()
    timing = (toc - tic)
    return preds, timing


# build datasets

# results
temp = [wd +"EdgeDeployment/pb_models/dnn/",
        wd +"EdgeDeployment/pb_models/c1nn/",
        wd +"EdgeDeployment/trt_models/dnn/",
        wd +"EdgeDeployment/trt_models/c1nn/",
        wd +"EdgeDeployment/FP16_models/dnn/",
        wd +"EdgeDeployment/FP16_models/c1nn/",
        ]

model_list = []
model_names = []
inf_time = []
inf_acc = []

for l in temp:
    temp = os.listdir(l)
    for i in temp:
        model_list.append(l + i)
        model_names.append(i[:-3])

# inference data
with open("y_test.p","rb") as f:
    test_y_test = pickle.load(f)
test_y_test = np.concatenate([test_y_test, test_y_test, test_y_test, 
                              test_y_test, test_y_test, test_y_test,
                              test_y_test, test_y_test, test_y_test,
                              test_y_test])


with open("X_test.p","rb") as f:
    test_X_test = pickle.load(f)
test_X_test = np.concatenate([test_X_test, test_X_test, test_X_test,
                              test_X_test, test_X_test, test_X_test, 
                              test_X_test, test_X_test, test_X_test, 
                              test_X_test])

#TODO: Capture data
for m in model_list:
    model = tf.keras.models.load_model(m)
    preds, timing = make_preds(model, test_X_test)
    y_preds = [np.argmax(x) for x in preds] 
    acc = []
    for i in range(len(test_y_test)):
        if y_preds[i] == test_y_test[i]:
            acc.append(1)
        else:
            acc.append(0)
    accuracy = sum(acc)/len(acc)
    inf_time.append(timing)
    inf_acc.append(accuracy)
    print(accuracy, timing)

"""
ap = argparse.ArgumentParser()
ap.add_argument("-pf", "--picklefile", required=True,
	help="X_test.p or X_test_c1nn.p file with path")
ap.add_argument("-p", "--path", required=True,
	help="path to location of model files and output")
args = vars(ap.parse_args())

os.chdir('/home/msweber/Code/')
args = {'picklefile': 'EdgeDeployment/data.p', 'path': 'EdgeDeployment/'}
Xpath = args["picklefile"]
print(str(Xpath))
       
with open(str(Xpath), "rb") as f:
    data = pickle.load(f)
path = args["path"]
print(path)
    
X_train, y_train, X_test, y_test = data  



model = tf.keras.models.load_model("/home/msweber/Code/EdgeDeployment/pb_models/dnn_128_3L.pb")
preds = model.predict(X_test)
y_preds = [np.argmax(x) for x in preds] 
acc = []
for i in range(len(y_test)):
    if y_preds[i] == y_test[i]:
        acc.append(1)
    else:
        acc.append(0)
accuracy = sum(acc)/len(acc)
    
print(y_preds)
"""        

