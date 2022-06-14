import onnx

# Load the ONNX model
model = onnx.load("mobile_own_0613_7777.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a Human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
print("ONNX model loaded successfully!")