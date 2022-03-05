#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 12:43:05 2022

@author: msweber
"""

from tensorflow.python.compiler.tensorrt import trt_convert as trt
#import tensorflow as tf
#from Model_Builder import Models
import os



class Converter():
    
    def __init__(self):
        pass
    
    def with_q(self, saved_model_path, output_path, precision="INT8"):
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
        conversion_params = conversion_params._replace(precision_mode=precision)
        converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_path,
                                            conversion_params=conversion_params)
        converter.convert()
        converter.save(output_path)
        



    def without_q(self, saved_model_path, output_path):
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
        converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_path,
                                            conversion_params=conversion_params)
        converter.convert()
        converter.save(output_path)
        
    
    def convert_dnn(self, model_name):
        """
        print("CONVERTING " + model_file[:-3] + " TO INT8 MODEL..............")
        self.with_q("pb_models/" + model_file,
                    "INT8_models/" + model_file, precision = "INT8")
        """
        print("CONVERTING " + model_name[:-3] + " TO FP16 MODEL..............")
        self.with_q("pb_models/dnn/" + model_name,
                    "FP16_models/dnn/" + model_name, precision = "FP16")
        print("CONVERTING " + model_name[:-3] + " TO TRT MODEL...............")
        self.without_q("pb_models/dnn/" + model_name, "trt_models/dnn/" + model_name)

    def convert_c1nn(self, model_name):
          """
          print("CONVERTING " + model_file[:-3] + " TO INT8 MODEL..............")
          self.with_q("pb_models/" + model_file,
                      "INT8_models/" + model_file, precision = "INT8")
          """
          print("CONVERTING " + model_name[:-3] + " TO FP16 MODEL..............")
          self.with_q("pb_models/c1nn/" + model_name,
                      "FP16_models/c1nn/" + model_name, precision = "FP16")
          print("CONVERTING " + model_name[:-3] + " TO TRT MODEL...............")
          self.without_q("pb_models/c1nn/" + model_name, "trt_models/c1nn/" + model_name)      
    
        
        
if __name__ == "__main__":
    """
    models = Models(784)
    model = models.build_dnn_1024_12L()
    model.load_weights("models/c1nn_1024_12L_best_model.h5")
    tf.saved_model.save(model, "models/trt_test.pb")
    model2 = tf.keras.models.load_model("models/trt_test.pb")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_test = X_test.reshape(-1,28*28)/ 255.0
    test = model2.predict(y_train)
    """
    
    conv = Converter()
    to_convert = os.listdir("pb_models/dnn/")
    for m in to_convert:
        conv.convert_dnn(m)
    to_convert = os.listdir("pb_models/c1nn/")
    for m in to_convert:
        conv.convert_c1nn(m)

        
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(precision_mode="FP16")
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir="dnn/dnn_512_6L.pb/saved_model.pb",
    conversion_params=conversion_params)
converter.convert()
def my_input_fn():
# Input for a single inference call, for a network that has two input tensors:
  Inp1 = X_train[0]
  yield (inp1)
converter.build(input_fn=my_input_fn)
converter.save("~/Code/EdgeDeployment/INT8_models/dnn/dnn_512_6L.pb")

saved_model_loaded = tf.saved_model.load(
    output_saved_model_dir, tags=[tag_constants.SERVING])
graph_func = saved_model_loaded.signatures[
    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
output = graph_func(input_data)[0].numpy()

