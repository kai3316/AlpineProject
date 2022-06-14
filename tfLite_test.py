import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="mobilenet_v2_random.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)



# get_tensor() returns a copy of the tensor data
# use tensor() in order to get a pointer to the tensor
# repeat 100 times to get a inferrence time of about 1 second
import time
start = time.time()
for i in range(100):
    interpreter.invoke()
    # output_data = interpreter.get_tensor(output_details[0]['index'])
end = time.time()
time = end - start
# fps
print("FPS: ", 100 / time)
# print(output_data)