#!/usr/bin/env python3
"""
Diagnostic script to test vLLM server startup
"""

import subprocess
import time
import requests
import sys
import os

def check_gpu():
    """Check GPU availability"""
    print("Checking GPU...")
    result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ No GPU detected or nvidia-smi not available")
        return False
    print("✅ GPU detected")
    return True

def check_vllm_installation():
    """Check if vLLM is installed"""
    print("\nChecking vLLM installation...")
    try:
        import vllm
        print(f"✅ vLLM version: {vllm.__version__}")
        return True
    except ImportError:
        print("❌ vLLM not installed. Install with: pip install vllm")
        return False

def check_model_path(model_name):
    """Check if model exists locally"""
    print(f"\nChecking model: {model_name}")
    # Check if it's a local path
    if os.path.exists(model_name):
        print(f"✅ Local model found at: {model_name}")
        return True
    
    # Check HuggingFace cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_id = model_name.replace("/", "--")
    model_path = os.path.join(cache_dir, f"models--{model_id}")
    
    if os.path.exists(model_path):
        print(f"✅ Model found in HuggingFace cache")
        return True
    
    print(f"⚠️  Model not found locally. Will download from HuggingFace.")
    return True  # vLLM will download it

def test_simple_server_startup(model_name="facebook/opt-125m", port=8000):
    """Test server startup with a small model"""
    print(f"\nTesting vLLM server startup with small model: {model_name}")
    print("This may take a few minutes on first run...")
    
    # Kill any existing processes on the port
    subprocess.run(f"lsof -ti:{port} | xargs kill -9 2>/dev/null", shell=True)
    time.sleep(2)
    
    # Start server
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--gpu-memory-utilization", "0.5",
        "--max-model-len", "512"  # Small context for testing
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    # Start server process
    with open('test_server_stdout.log', 'w') as stdout_file, \
         open('test_server_stderr.log', 'w') as stderr_file:
        
        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file
        )
    
    # Wait for server to start
    print("Waiting for server to start...")
    max_wait = 120  # 2 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        # Check if process is still running
        if process.poll() is not None:
            print("❌ Server process terminated unexpectedly")
            print("Check test_server_stderr.log for errors")
            with open('test_server_stderr.log', 'r') as f:
                print("\nError output:")
                print(f.read())
            return False
        
        # Try to connect
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=1)
            if response.status_code == 200:
                print(f"✅ Server started successfully after {time.time() - start_time:.1f} seconds")
                
                # Test inference
                print("\nTesting inference...")
                payload = {
                    "model": model_name,
                    "prompt": "Hello, world!",
                    "max_tokens": 10
                }
                
                response = requests.post(
                    f"http://localhost:{port}/v1/completions",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    print("✅ Inference test successful")
                    result = response.json()
                    print(f"Response: {result['choices'][0]['text'][:50]}...")
                else:
                    print(f"❌ Inference failed: {response.status_code}")
                    print(response.text)
                
                # Cleanup
                process.terminate()
                process.wait()
                return True
                
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(2)
    
    print(f"❌ Server failed to start after {max_wait} seconds")
    process.terminate()
    process.wait()
    return False

def check_memory():
    """Check available GPU memory"""
    print("\nChecking GPU memory...")
    result = subprocess.run(
        "nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits",
        shell=True, capture_output=True, text=True
    )
    
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for i, line in enumerate(lines):
            total, free = map(int, line.split(', '))
            used = total - free
            print(f"GPU {i}: {free}MB free / {total}MB total (Used: {used}MB)")
            
            if free < 5000:  # Less than 5GB free
                print(f"⚠️  Low GPU memory on GPU {i}. This might cause issues with larger models.")
    
def diagnose_full_model(model_name, port=8000):
    """Diagnose issues with the full model"""
    print(f"\n{'='*60}")
    print(f"Diagnosing: {model_name}")
    print(f"{'='*60}")
    
    # Estimate memory requirement
    if "7b" in model_name.lower():
        print("⚠️  This is a 7B model. It requires approximately 14-16GB of GPU memory.")
    elif "13b" in model_name.lower():
        print("⚠️  This is a 13B model. It requires approximately 26-30GB of GPU memory.")
    
    print("\nTrying to start server with your model...")
    print("Check test_server_stderr.log for detailed error messages")
    
    # Try with minimal settings
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--gpu-memory-utilization", "0.9",
        "--max-model-len", "1024",  # Reduced context
        "--dtype", "half"  # Use fp16 to save memory
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    with open('test_server_stdout.log', 'w') as stdout_file, \
         open('test_server_stderr.log', 'w') as stderr_file:
        
        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file
        )
    
    # Wait briefly
    time.sleep(30)
    
    if process.poll() is not None:
        print("❌ Server failed to start")
        with open('test_server_stderr.log', 'r') as f:
            errors = f.read()
            if "CUDA out of memory" in errors or "OutOfMemoryError" in errors:
                print("❌ Out of GPU memory. Try:")
                print("   - Using a smaller model")
                print("   - Reducing max-model-len")
                print("   - Using tensor parallelism if you have multiple GPUs")
                print("   - Using quantization (--quantization awq)")
            elif "404" in errors or "not found" in errors:
                print("❌ Model not found. Check the model name or path")
            else:
                print("Error details (last 20 lines):")
                print('\n'.join(errors.split('\n')[-20:]))
    else:
        print("⚠️  Server process is running. Checking health...")
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is healthy!")
        except:
            print("⚠️  Server is starting slowly. This is normal for large models.")
            print("    It may take 2-5 minutes for first-time model loading.")
    
    process.terminate()
    process.wait()

def main():
    print("vLLM Diagnostic Tool")
    print("=" * 60)
    
    # Basic checks
    if not check_gpu():
        sys.exit(1)
    
    if not check_vllm_installation():
        sys.exit(1)
    
    check_memory()
    
    # Test with small model first
    if test_simple_server_startup():
        print("\n✅ Basic vLLM functionality is working!")
        
        # Now test with the actual model
        model_name = input("\nEnter your model name (or press Enter for default): ").strip()
        if not model_name:
            model_name = "meta-llama/Llama-2-7b-hf"
        
        check_model_path(model_name)
        diagnose_full_model(model_name)
    else:
        print("\n❌ Basic vLLM test failed. Check the logs for details.")
        print("\nCommon issues:")
        print("1. Incompatible CUDA version")
        print("2. Corrupted vLLM installation")
        print("3. Insufficient GPU memory even for small models")
        print("4. Missing dependencies")
        
    print("\n" + "=" * 60)
    print("Diagnostic complete. Check log files for more details:")
    print("- test_server_stdout.log")
    print("- test_server_stderr.log")

if __name__ == "__main__":
    main()