import tensorflow as tf

model = tf.saved_model.load("/home/kai/Desktop/AlpineProject/mbv2_ca0613")
model.trainable = False

input_tensor = tf.random.uniform([1, 3, 224, 224])
out = model(**{'input.1': input_tensor})
print(out)