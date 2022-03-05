# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import datasets, optimizers
import time
import numpy as np
from new_model_builder import Models
# from tensorflow.python.compiler.tensorrt import trt_convert

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

def train():
    # You can directly use datasets.mnist.load_data(), if the network is good, you can connect to the external network,
    # If you can’t download, you can download the file yourself
    (x, y), (x_val, y_val) = datasets.mnist.load_data()
    print('datasets:', x.shape, y.shape, x.min(), x.max())

    db = tf.data.Dataset.from_tensor_slices((x, y))
    db = db.map(preprocess).shuffle(10000).batch(batchsz)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.map(preprocess).batch(batchsz)
    """
    # sample = next(iter(db))
    # print(sample[0].shape, sample[1].shape)
    inputs = tf.keras.Input(shape=(28,28,1), name='input')
    # [28, 28, 1] => [28, 28, 64]
    input = tf.keras.layers.Flatten(name="flatten")(inputs)
    fc_1 = tf.keras.layers.Dense(512, activation='relu', name='fc_1')(input)
    fc_2 = tf.keras.layers.Dense(256, activation='relu', name='fc_2')(fc_1)
    pred = tf.keras.layers.Dense(10, activation='softmax', name='output')(fc_2)

    model = tf.keras.Model(inputs=inputs, outputs=pred, name='mnist')
    """ 
    models = Models()
    model = models.build_dnn_128_3L() 
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
    model.save('tf_models/tf_savedmodel', save_format='tf')
    # Method 2:
    # tf.saved_model.save(obj=model, export_dir="./model/")

if __name__ == "__main__":
    train()

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.compiler.tensorrt import trt_convert as trt
params = trt.DEFAULT_TRT_CONVERSION_PARAMS
params._replace(precision_mode=trt.TrtPrecisionMode.FP32)
converter = trt.TrtGraphConverterV2(input_saved_model_dir='tf_models/tf_savedmodel', conversion_params=params)
# Complete the conversion, but no optimization is performed at this time, the optimization is completed when the inference is performed
converter.convert()
converter.save("trt_models/dnn_512")


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices)> 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_test = x_test.astype('float32')
# x_test = x_test.reshape(10000, 784)
x_test/= 255

# Read model
saved_model_loaded = tf.saved_model.load("trt_models/dnn_512", tags=[trt.tag_constants.SERVING])
# Get the inference function, you can also use saved_model_loaded.signatures['serving_default']
graph_func = saved_model_loaded.signatures[trt.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
# Turn the variables in the model into constants. This step can be omitted, and it is also possible to directly call graph_func
frozen_func = trt.convert_to_constants.convert_variables_to_constants_v2(graph_func)

count = 20
for x,y in zip(x_test, y_test):
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