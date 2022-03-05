from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf
import numpy as np
import pickle
import time
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


with open("X_test.p","rb") as f:
    X_test = pickle.load(f)
with open("y_test.p","rb") as f:
    y_test = pickle.load(f)
    
# input_fn: a generator function that yields input data as a list or tuple,
# which will be used to execute the converted signature to generate TensorRT
# engines. Example:
def my_input_fn():
    # Let's assume a network with 2 input tensors. We generate 3 sets
    # of dummy input data:
    input_shapes = [[(1, 784)]] # input list
    for shapes in input_shapes:
        # return a list of input tensors
        yield [np.zeros(x).astype(np.float32) for x in shapes]
"""        
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Conversion Parameters 
conversion_params = trt.TrtConversionParams(
    precision_mode=trt.TrtPrecisionMode.FP32)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir="pb_models/dnn/dnn_128_6L.pb",
    conversion_params=conversion_params)

# Converter method used to partition and optimize TensorRT compatible segments
converter.convert()

# Optionally, build TensorRT engines before deployment to save time at runtime
# Note that this is GPU specific, and as a rule of thumb, we recommend building at runtime
converter.build(input_fn=my_input_fn)

# Save the model to the disk 
converter.save("INT8_models/dnn/dnn_128_6L")
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.compiler.tensorrt import trt_convert as trt
params = trt.DEFAULT_TRT_CONVERSION_PARAMS
params._replace(precision_mode=trt.TrtPrecisionMode.FP32)
converter = trt.TrtGraphConverterV2(input_saved_model_dir="pb_models/dnn/dnn_128_6L.pb", conversion_params=params)
# Complete the conversion, but no optimization is performed at this time, the optimization is completed when the inference is performed
converter.convert()
converter.save('INT8_models/dnn/dnn_128_6L')



from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_test = X_test.astype('float32')
# x_test = x_test.reshape(10000, 784)
X_test/= 255


# Read model
saved_model_loaded = tf.saved_model.load("INT8_models/dnn/dnn_128_6L", tags=[trt.tag_constants.SERVING])
# Get the inference function, you can also use saved_model_loaded.signatures['serving_default']
graph_func = saved_model_loaded.signatures[trt.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
# Turn the variables in the model into constants. This step can be omitted, and it is also possible to directly call graph_func
frozen_func = trt.convert_to_constants.convert_variables_to_constants_v2(graph_func)

count = 20
for x,y in zip(X_test, y_test):
    x = tf.cast(x, tf.float32)
    start = time.time()
    # frozen_func(x) The return value is a list
    # The list contains an element, which is the output tensor, which is converted to numpy format using .numpy()
    output = frozen_func(x)[0].numpy()
    end = time.time()
    times = (end-start) * 1000.0
    print("tensorrt times: ", times, "ms")
    result = np.argmax(output, 1)
    print("prediction result: ", result, "| ", "true result: ", y)

    if count == 0:
        break
    count -= 1
    

# Load converted model and infer
model = tf.saved_model.load("INT8_models/dnn/dnn_128_6L")
func = model.signatures['serving_default']
output = func(test_X_test)

