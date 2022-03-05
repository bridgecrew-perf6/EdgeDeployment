#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:33:22 2022

@author: msweber
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--datapath", required=True,
	help="path to the X_test.p file")
ap.add_argument("-p", "--path", required=True,
	help="path to location of model files and output")
args = vars(ap.parse_args())
print(args)



import tensorflow as tf

model = tf.saved_model.load("EdgeDeployment/trt_models/dnn_512_6L.pb")
func = root.signatures['serving_default']
output = func(input_tensor)