import torch

def check_gpu():
    # Check if PyTorch detects a GPU}
    print("PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA available: ✅  " )
        print("CUDA version:", torch.version.cuda)
        print("Number of GPUs:", torch.cuda.device_count())
        print("✅ GPU detected.")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Memory Capacity: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Test a simple operation on the GPU
        try:
            x = torch.rand(1000, 1000).cuda()  # Create a random tensor on the GPU
            y = torch.matmul(x, x)  # Perform a matrix multiplication
            print("✅ GPU operation completed successfully.")
        except Exception as e:
            print("❌ Error performing operations on the GPU.")
            print(e)
    else:
        print("❌ No GPU detected. Make sure drivers and CUDA are properly installed.")

if __name__ == "__main__":
    check_gpu()