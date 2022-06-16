import onnx
onnx_model = onnx.load('resnet50_random.onnx')
from onnx_tf.backend import prepare

tf_rep = prepare(onnx_model)
tf_rep.export_graph('resnet50_random')
print("ONNX model to tf model successfully!")