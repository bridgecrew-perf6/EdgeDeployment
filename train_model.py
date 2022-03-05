# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import datasets, optimizers
import pickle
import time
import numpy as np
from new_model_builder import Models
from contextlib import redirect_stdout
from tensorflow.python.compiler.tensorrt import trt_convert as trt

def model_select(model_name):
    if model_name == 'dnn_128_3L':
        M = Models()
        model = M.build_dnn_128_3L() 
        return model
    elif model_name == 'dnn_512_3L':
        M = Models()
        model = M.build_dnn_512_3L() 
        return model
    elif model_name == 'dnn_1024_3L':
        M = Models()
        model = M.build_dnn_1024_3L() 
        return model
    elif model_name == 'dnn_128_6L':
        M = Models()
        model = M.build_dnn_128_6L() 
        return model
    elif model_name == 'dnn_512_6L':
        M = Models()
        model = M.build_dnn_512_6L() 
        return model
    elif model_name == 'dnn_1024_6L':
        M = Models()
        model = M.build_dnn_1024_6L() 
        return model
    elif model_name == 'dnn_128_12L':
        M = Models()
        model = M.build_dnn_128_12L() 
        return model
    elif model_name == 'dnn_512_12L':
        M = Models()
        model = M.build_dnn_512_12L() 
        return model
    elif model_name == 'dnn_1024_12L':
        M = Models()
        model = M.build_dnn_1024_12L() 
        return model
    elif model_name == 'c1nn_32':
        M = Models()
        model = M.build_c1nn32() 
        return model
    elif model_name == 'c1nn_32_16':
        M = Models()
        model = M.build_c1nn32_16() 
        return model
    elif model_name == 'c1nn_64':
        M = Models()
        model = M.build_c1nn64() 
        return model
    elif model_name == 'c1nn_64_32':
        M = Models()
        model = M.build_c1nn64_32() 
        return model
    elif model_name == 'c1nn_64_32_16':
        M = Models()
        model = M.build_c1nn64_32_16() 
        return model
    elif model_name == 'c1nn_128_64_32':
        M = Models()
        model = M.build_c1nn128_64_32() 
        return model
    

def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.expand_dims(x, axis=-1)
    x = tf.cast(x, dtype=tf.float32)/255.
    # x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


batchsz = 128

def train(model_name):
    # You can directly use datasets.mnist.load_data(), if the network is good, you can connect to the external network,
    # If you can’t download, you can download the file yourself
    (x, y), (x_val, y_val) = datasets.mnist.load_data()
    print('datasets:', x.shape, y.shape, x.min(), x.max())

    db = tf.data.Dataset.from_tensor_slices((x, y))
    db = db.map(preprocess).shuffle(10000).batch(batchsz)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.map(preprocess).batch(batchsz)

    
    model = model_select(model_name) 

    with open("model_summaries/" + model_name + ".txt", "w") as f:
        with redirect_stdout(f):
            model.summary()
    model.summary()
    Loss = []
    Acc = []
    optimizer = optimizers.Adam(0.001)
    # epoches = 5
    for epoch in range(5):
        # Create parameters for testing accuracy
        total_num = 0
        total_correct = 0
        for step, (x,y) in enumerate(db):
            with tf.GradientTape() as tape:

                pred = model(x)
                loss = tf.keras.losses.categorical_crossentropy(y_pred=pred,
                                                                y_true=y,
                                                                from_logits=False)
                loss = tf.reduce_mean(loss)
                grades = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grades, model.trainable_variables))
                # Output loss value
            if step% 10 == 0:
                print("epoch: ", epoch, "step: ", step, "loss: ", loss.numpy())
                Loss.append(loss)

        # Calculate the accuracy, convert the output of the fully connected layer into a probability value output
        for step, (x_val, y_val) in enumerate(ds_val):
            # Predict the output of the test set

            pred = model(x_val)
            # pred = tf.nn.softmax(pred, axis=1)
            pred = tf.argmax(pred, axis=1)
            pred = tf.cast(pred, tf.int32)
            y_val = tf.argmax(y_val, axis=1)
            y_val = tf.cast(y_val, tf.int32)
            correct = tf.equal(pred, y_val)
            correct = tf.cast(correct, tf.int32)
            correct = tf.reduce_sum(correct)
            total_correct += int(correct)
            total_num += x_val.shape[0]
            if step% 20 == 0:
                acc_step = total_correct/total_num
                print(str(step) + "The stage precision of the step is:", acc_step)
                Acc.append(float(acc_step))

        acc = total_correct/total_num
        print("epoch %d test acc: "% epoch, acc)
    # Method 1:
    model.save("pb_models/" + model_name + ".pb", save_format='tf')
    # Method 2:
    # tf.saved_model.save(obj=model, export_dir="./model/")


if __name__ == "__main__":
    model_list = ['dnn_128_3L', 'dnn_512_3L', 'dnn_1024_3L', 'dnn_128_6L',
                  'dnn_512_6L', 'dnn_1024_6L', 'dnn_128_12L', 'dnn_512_12L', 
                  'dnn_1024_12L', 'c1nn_32', 'c1nn_32_16', 'c1nn_64',
                  'c1nn_64_32', 'c1nn_64_32_16', 'c1nn_128_64_32']
    
    tf_model = []
    trt_model = []
    FP16_model = []
    INT8_model = []
    combined = []
    
    for model_name in model_list:
        tf_model.append('pb_models/' + model_name + '.pb')
        trt_model.append('trt_models/' + model_name + '.pb')
        FP16_model.append('FP16_models/' + model_name + '.pb')
        INT8_model.append('INT8_models/' + model_name + '.pb')
        
    
    with open("model_lists.p", "wb") as f:
        pickle.dump([tf_model, trt_model, FP16_model, INT8_model], f)
    
    train(model_name)
        
        
for model_name in model_list:
    train(model_name)
 
for i in range(len(tf_model)):
    ### TRT MODEL #############################################################
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    params._replace(precision_mode=trt.TrtPrecisionMode.FP32)
    
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=tf_model[i], 
                                        conversion_params=params)
    # Complete the conversion, but no optimization is performed at this time, 
    #the optimization is completed when the inference is performed
    converter.convert()
    converter.save(trt_model[i])
    
    ### FP16 MODEL ############################################################
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    params._replace(precision_mode=trt.TrtPrecisionMode.FP16)
    
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=tf_model[i], 
                                        conversion_params=params)
    # Complete the conversion, but no optimization is performed at this time, 
    #the optimization is completed when the inference is performed
    converter.convert()
    converter.save(FP16_model[i])
    
    ### INT8 MODEL ############################################################
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    params._replace(precision_mode=trt.TrtPrecisionMode.INT8)
    
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=tf_model[i], 
                                        conversion_params=params)
    # Complete the conversion, but no optimization is performed at this time, 
    #the optimization is completed when the inference is performed
    converter.convert()
    converter.save(INT8_model[i])






