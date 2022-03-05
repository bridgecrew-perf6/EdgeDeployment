#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 12:06:28 2022

@author: msweber
"""
import os
import socket
host = socket.gethostname()
import pickle
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

with open("data.p", "rb") as f:
    (x_train, y_train), (x_test, y_test) = pickle.load(f)

with open("model_lists.p", "rb") as f:
    model_lists = pickle.load(f)
 
combined = []
for z in model_lists:
    combined.extend(z)

RESULTS = {}
for m_path in combined:
    print(m_path)
        
    # Read model
    saved_model_loaded = tf.saved_model.load(m_path, tags=[trt.tag_constants.SERVING])
    # Get the inference function, you can also use saved_model_loaded.signatures['serving_default']
    graph_func = saved_model_loaded.signatures[trt.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    # Turn the variables in the model into constants. This step can be omitted, and it is also possible to directly call graph_func
    frozen_func = trt.convert_to_constants.convert_variables_to_constants_v2(graph_func)
    
    correct = []
    timings = []
    count = 200
    for x,y in zip(x_test, y_test):
        x = tf.cast(x, tf.float32)
        start = time.time()
        # frozen_func(x) The return value is a list
        # The list contains an element, which is the output tensor, which is converted to numpy format using .numpy()
        output = frozen_func(x)[0].numpy()
        end = time.time()
        times = (end-start) * 1000.0
        timings.append(times)
        print("tensorrt times: ", times, "ms")
        result = np.argmax(output, 1)
        if result == y:
            correct.append(1)
        else:
            correct.append(0)
        print("prediction result: ", result, "| ", "true result: ", y)
    
        if count == 0:
            break
        count -= 1
        
    avg_time= np.mean(timings)
    accuracy = np.mean(correct)
    print(avg_time, accuracy)
    RESULTS.update({m_path: (avg_time, accuracy)})
       
with open(host + "_results.p", "wb") as f:
    pickle.dump(RESULTS, f)