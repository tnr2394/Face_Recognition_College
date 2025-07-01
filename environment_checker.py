import os
import subprocess
import platform

def find_file_in_path(filename, search_path):
    """Helper function to find a file in a given directory path."""
    for root, _, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def check_nvidia_smi():
    """Checks for nvidia-smi and reports driver and CUDA versions."""
    print("--- 1. Checking for NVIDIA Drivers ---")
    try:
        if platform.system() == "Windows":
            cmd = ["nvidia-smi.exe"]
        else:
            cmd = ["nvidia-smi"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ `nvidia-smi` found. NVIDIA drivers are installed.")
        # Parse output to find driver and supported CUDA version
        if "Driver Version" in result.stdout:
            driver_version = result.stdout.split("Driver Version: ")[1].split()[0]
            print(f"   Driver Version: {driver_version}")
        if "CUDA Version" in result.stdout:
            cuda_version = result.stdout.split("CUDA Version: ")[1].split()[0]
            print(f"   Highest Supported CUDA Version by Driver: {cuda_version}")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ `nvidia-smi` not found.")
        print("   ACTION: Please install the latest NVIDIA drivers for your GPU from https://www.nvidia.com/Download/index.aspx")
        return False

def check_cuda_toolkit():
    """Checks for the CUDA Toolkit installation via CUDA_PATH."""
    print("\n--- 2. Checking for CUDA Toolkit ---")
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path and os.path.exists(cuda_path):
        print(f"✅ CUDA Toolkit found at: {cuda_path}")
        # Check for nvcc compiler as a secondary verification
        nvcc_path = os.path.join(cuda_path, 'bin', 'nvcc.exe' if platform.system() == "Windows" else "nvcc")
        if os.path.exists(nvcc_path):
            print("   ✅ `nvcc` compiler found.")
            return cuda_path
        else:
            print("   ⚠️ `nvcc` compiler not found in the bin directory. The installation might be corrupt.")
            return None
    else:
        print("❌ CUDA Toolkit not found.")
        print("   The `CUDA_PATH` environment variable is not set or points to a non-existent directory.")
        print("   ACTION: Please install the CUDA Toolkit 12.1 from: https://developer.nvidia.com/cuda-12-1-0-download-archive")
        return None

def check_cudnn(cuda_path):
    """Checks for cuDNN library files within the CUDA Toolkit directory."""
    print("\n--- 3. Checking for cuDNN ---")
    if not cuda_path:
        print("   Skipping cuDNN check because CUDA Toolkit was not found.")
        return False

    cudnn_found = False
    if platform.system() == "Windows":
        # In Windows, the key file is cudnn64_*.dll in the bin folder
        bin_path = os.path.join(cuda_path, 'bin')
        if os.path.exists(bin_path):
            for filename in os.listdir(bin_path):
                if filename.startswith('cudnn64_') and filename.endswith('.dll'):
                    print(f"✅ cuDNN library found: {filename}")
                    cudnn_found = True
                    break
    else: # Linux
        lib_path = os.path.join(cuda_path, 'lib64')
        if os.path.exists(lib_path):
            for filename in os.listdir(lib_path):
                if 'libcudnn.so' in filename:
                    print(f"✅ cuDNN library found: {filename}")
                    cudnn_found = True
                    break

    if not cudnn_found:
        print("❌ cuDNN library not found in the CUDA Toolkit directory.")
        print("   ACTION: Please download cuDNN v8.9.7 for CUDA 12.x from: https://developer.nvidia.com/rdp/cudnn-archive")
        print("   Then, copy the contents of the unzipped 'bin', 'include', and 'lib' folders into your CUDA Toolkit directory.")
        print(f"   (Your CUDA Toolkit directory is: {cuda_path})")
    
    return cudnn_found

if __name__ == "__main__":
    print("=================================================")
    print("  Running GPU Environment Configuration Checker")
    print("=================================================")
    drivers_ok = check_nvidia_smi()
    if drivers_ok:
        cuda_path = check_cuda_toolkit()
        if cuda_path:
            check_cudnn(cuda_path)

    print("\n--- Summary ---")
    print("If all checks above are marked with ✅, your GPU environment is likely ready.")
    print("If you see any ❌, please follow the ACTION steps provided.")
    print("After making changes, please restart your computer before testing again.")
