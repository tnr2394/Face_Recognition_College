import onnx

model = onnx.load("OFIQ-MODELS/models/unified_quality_score/magface_iresnet50_norm.onnx")
print("Model IR version:", model.ir_version)
