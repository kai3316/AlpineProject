import tensorflow as tf
# Convert the model
tf_model_path = "/home/kai/Desktop/AlpineProject/resnet50_random"
tflite_model_path = '/home/kai/Desktop/AlpineProject/resnet50_random_fp16.tflite'
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save the model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
print("TFLite model saved to: %s" % tflite_model_path)