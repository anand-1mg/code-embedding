import torch

from salesforce.sfr_code_embedding_400 import run_benchmark

if __name__ == "__main__":
    # Set CUDA device options for best performance
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
        torch.backends.cudnn.allow_tf32 = True

    # Print system info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Adjust these parameters based on your g5.xlarge instance
    results = run_benchmark(
        batch_size=64,  # Try 32, 64, 128 to find optimal
        max_workers=4,  # g5.xlarge has 4 vCPUs
        max_codes=1000,  # Number of code snippets to process
        max_length=1024  # Max token length
    )

    # Optional: Try different configurations to find optimal settings
    # batch_sizes = [32, 64, 128]
    # for bs in batch_sizes:
    #     run_benchmark(batch_size=bs)