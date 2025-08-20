#!/usr/bin/env python3

import torch
import time
from ultralytics import YOLO
import os
from pathlib import Path

def test_gpu_acceleration():
    print("🔍 Testing GPU Acceleration on Mac M3")
    print("=" * 50)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check device availability
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Determine optimal device
    if torch.backends.mps.is_available():
        device = "mps"
        print("🚀 Using MPS (Metal Performance Shaders) for Apple Silicon")
    elif torch.cuda.is_available():
        device = "cuda"
        print("🚀 Using CUDA GPU")
    else:
        device = "cpu"
        print("⚠️ Using CPU (no GPU acceleration)")
    
    # Test with a simple tensor operation
    print(f"\n🧪 Testing tensor operations on {device}...")
    
    # Create test tensor
    x = torch.randn(1000, 1000)
    
    # Time CPU operation
    start_time = time.time()
    cpu_result = torch.mm(x, x)
    cpu_time = time.time() - start_time
    print(f"CPU matrix multiplication: {cpu_time:.4f} seconds")
    
    if device != "cpu":
        # Time GPU operation
        x_gpu = x.to(device)
        start_time = time.time()
        gpu_result = torch.mm(x_gpu, x_gpu)
        gpu_time = time.time() - start_time
        print(f"{device.upper()} matrix multiplication: {gpu_time:.4f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"Speedup: {speedup:.2f}x faster on {device.upper()}")
    
    # Test YOLO model loading
    print(f"\n🤖 Testing YOLO model loading on {device}...")
    
    weights_dir = Path("weights")
    if weights_dir.exists():
        weight_files = list(weights_dir.glob("*.pt"))
        if weight_files:
            test_weight = str(weight_files[0])
            print(f"Testing with weight: {test_weight}")
            
            try:
                # Load model
                start_time = time.time()
                model = YOLO(test_weight)
                model.to(device)
                load_time = time.time() - start_time
                print(f"Model loaded in {load_time:.4f} seconds")
                
                # Test inference
                print("Testing inference...")
                import numpy as np
                test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                
                start_time = time.time()
                results = model(test_image, device=device, verbose=False)
                inference_time = time.time() - start_time
                print(f"Inference completed in {inference_time:.4f} seconds")
                
                print("✅ GPU acceleration test completed successfully!")
                
            except Exception as e:
                print(f"❌ Error testing YOLO model: {str(e)}")
        else:
            print("⚠️ No weight files found in weights directory")
    else:
        print("⚠️ Weights directory not found")
    
    print("\n" + "=" * 50)
    print("🎉 GPU acceleration test completed!")

if __name__ == "__main__":
    test_gpu_acceleration() 