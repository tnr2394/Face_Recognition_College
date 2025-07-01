import torch
import onnxruntime as ort
import numpy as np
import os

def check_dependencies():
    """Checks for the presence of onnx."""
    print("--- Checking Dependencies ---")
    try:
        import onnx
        import onnx.helper
        print("✅ `onnx` package is installed.")
        return True
    except ImportError:
        print("❌ `onnx` package is not installed. Please run: pip install onnx")
        print("   Cannot perform the ONNX Runtime GPU test without it.")
        return False

def check_pytorch():
    """Checks PyTorch GPU availability and prints details."""
    print("\n--- Checking PyTorch ---")
    if torch.cuda.is_available():
        print("✅ PyTorch has access to the GPU.")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        print(f"   Current GPU name: {torch.cuda.get_device_name(0)}")
        try:
            # Simple test operation
            tensor = torch.tensor([1.0, 2.0]).to('cuda')
            print("✅ Successfully created a tensor on the GPU.")
            print(f"   Test tensor: {tensor}")
        except Exception as e:
            print(f"❌ Failed to perform a test operation on the GPU: {e}")
    else:
        print("❌ PyTorch cannot find a compatible GPU.")
        print("   Possible reasons:")
        print("   1. NVIDIA drivers are not installed or are outdated.")
        print("   2. The installed version of PyTorch does not match the CUDA version on your system.")
        print("   3. You have a CPU-only version of PyTorch installed.")

def check_onnxruntime():
    """Checks ONNX Runtime GPU availability using the actual OFIQ model."""
    print("\n--- Checking ONNX Runtime ---")
    available_providers = ort.get_available_providers()
    print(f"   Available ONNX Runtime providers: {available_providers}")
    
    model_path = "OFIQ-MODELS/models/unified_quality_score/magface_iresnet50_norm.onnx"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        print("   Skipping ONNX Runtime GPU test.")
        return

    if 'CUDAExecutionProvider' in available_providers:
        print("✅ ONNX Runtime has access to the GPU (CUDAExecutionProvider is available).")
        try:
            print(f"   Attempting to load model: {model_path}")
            sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
            
            input_name = sess.get_inputs()[0].name
            # The model expects a (1, 3, 112, 112) tensor
            input_shape = (1, 3, 112, 112)
            dummy_input = np.random.rand(*input_shape).astype(np.float32)

            print("   Running test inference with the OFIQ model on the GPU...")
            result = sess.run(None, {input_name: dummy_input})
            
            print("✅ Successfully ran a test inference with the OFIQ model on the GPU.")
            print(f"   Output score shape: {result[0].shape}")
            print(f"   Output score (sample): {result[0][0].item():.4f}")

        except Exception as e:
            print(f"❌ Failed to run a test inference with the OFIQ model on the GPU: {e}")
            print("   This might happen if there's a mismatch between CUDA, cuDNN, and the onnxruntime-gpu package,")
            print("   or an incompatibility with the model itself.")

    else:
        print("❌ ONNX Runtime cannot find the CUDAExecutionProvider.")
        print("   Possible reasons:")
        print("   1. You have the `onnxruntime` package installed, not `onnxruntime-gpu`.")
        print("   2. There is a version mismatch with your CUDA installation.")


if __name__ == "__main__":
    print("===================================")
    print("  Running GPU Configuration Test")
    print("===================================")
    dependencies_ok = check_dependencies()
    check_pytorch()
    if dependencies_ok:
        check_onnxruntime()
    else:
        print("\nSkipping ONNX Runtime GPU test due to missing `onnx` package.")
