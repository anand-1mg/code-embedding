import math
import time

import pandas as pd
import torch
import concurrent.futures
from transformers import AutoModel, AutoTokenizer
from typing import List, Union

from constants.constants import ModelCheckPoints

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
MAX_CHARACTER_SIZE = 8192


class SfrCodeEmbedding400:
    def __init__(self, checkpoint: str, max_length: int = MAX_CHARACTER_SIZE):
        print(f"Loading model from checkpoint: {checkpoint}")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, add_eos_token=True)
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

        # Optimize model for inference
        self.model.eval()
        if device == "cuda":
            self.model = self.model.half()  # Use FP16 for faster inference on GPU
            torch.backends.cudnn.benchmark = True  # Optimize CUDA operations

    def create_embedding(self, inputs: Union[str, List[str]]) -> torch.Tensor:
        """Create embeddings for a single input or a list of inputs"""
        if isinstance(inputs, str):
            inputs = [inputs]

        with torch.cuda.amp.autocast():  # Enable automatic mixed precision
            tokens = self.tokenizer(inputs, return_tensors="pt", padding=True,
                                    truncation=True, max_length=self.max_length).to(device)
            with torch.no_grad():  # Disable gradients for inference
                embeddings = self.model(**tokens).last_hidden_state

        return embeddings

    def process_batches(self, code_snippets: List[str], batch_size: int = 64) -> List[torch.Tensor]:
        """Process code snippets in optimized batches"""
        results = []
        num_batches = len(code_snippets) // batch_size + (1 if len(code_snippets) % batch_size > 0 else 0)

        for i in range(num_batches):
            batch = code_snippets[i * batch_size:(i + 1) * batch_size]
            embeddings = self.create_embedding(batch)
            results.append(embeddings)
            print(f"Processed batch {i + 1}/{num_batches}")

            # Optional: clear cache between large batches
            if i % 10 == 0 and device == "cuda":
                torch.cuda.empty_cache()

        return results

    def process_in_parallel(self, code_snippets: List[str], batch_size: int = 10, max_workers: int = None) -> List[
        torch.Tensor]:
        results = []
        num_batches = len(code_snippets) // batch_size + (1 if len(code_snippets) % batch_size > 0 else 0)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Set sensible default max_workers
        if max_workers is None:
            max_workers = 1 if device == "cuda" else min(max_workers, math.floor(torch.get_num_threads() * 0.5))

        print(f"Processing with {max_workers} workers")

        # Select executor type based on device
        executor_cls = concurrent.futures.ThreadPoolExecutor if device == "cuda" else concurrent.futures.ProcessPoolExecutor
        executor_cls = concurrent.futures.ThreadPoolExecutor
        with executor_cls(max_workers=max_workers) as executor:
            futures = []
            for i in range(num_batches):
                batch = code_snippets[i * batch_size:(i + 1) * batch_size]
                futures.append(executor.submit(self.create_embedding, batch))

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                results.append(future.result())
                print(f"Completed batch {i + 1}/{num_batches}")

        return results


def get_code_search_code_snippets():
    code_search_net_dataset_path = "benchmarks/codesearchnet/data/csn_1lakh.csv"
    records = pd.read_csv(code_search_net_dataset_path)

    # Get all values from the 'code' column
    code_snippets = records['code'].tolist()

    return code_snippets


def get_test_code_records(count: int, static: bool = False):
    # Example input code
    if static:
        code = """
            class PRDiffHandler:
            def __init__(self, pr_service: BasePR):
                self.pr_service = pr_service
                self.pr_diff = None  # Original PR diff content
                self.pr_diff_mappings = {}  # Maps operation/agent to indices in pr_diffs
                self.pr_diffs = []  # List of unique PR diffs for efficient memory usage
    
            async def get_effective_pr_diff(self, operation: str = "code_review", agent_id: str = None) -> str:
                if not self.pr_diff:
                    self.pr_diff = await self.pr_service.get_commit_diff_or_pr_diff()
                # If no mappings exist, populate them
                if not self.pr_diff_mappings:
                    self.set_pr_diff_mappings(operation)
    
                # Choose the correct PR diff based on the operation and agent ID
                if operation == "chat":
                    diff_index = self.pr_diff_mappings.get("chat")
                else:
                    if agent_id:
                        diff_index = self.pr_diff_mappings[agent_id]
                    else:
                        diff_index = self.pr_diff_mappings["global_diff"]
    
                # Return the corresponding PR diff from the list
                return self.pr_diffs[diff_index]
            """

        code_snippets = [code] * count
    else:
        code_snippets = get_code_search_code_snippets()[:count]
    return code_snippets


def run_benchmark(batch_size: int = 64, max_workers: int = 4, max_codes: int = 1000, max_length: int = MAX_CHARACTER_SIZE):
    """Run a benchmark to measure QPS with different configurations"""
    print(f"Starting benchmark with batch_size={batch_size}, max_workers={max_workers}, max_codes={max_codes}")

    # Initialize model
    sfr_embedder = SfrCodeEmbedding400(ModelCheckPoints.SFR_SMALL.value, max_length=max_length)

    code_snippets = get_test_code_records(max_codes, static=True)

    # GPU warmup
    print("Warming up GPU...")
    _ = sfr_embedder.create_embedding(code_snippets[:min(10, len(code_snippets))])

    # Run parallel processing benchmark
    print("\nRunning parallel processing benchmark...")
    start_time = time.time()
    _ = sfr_embedder.process_in_parallel(code_snippets, batch_size=batch_size, max_workers=max_workers)
    end_time = time.time()

    # Calculate and print results
    total_time = end_time - start_time
    qps = max_codes / total_time
    print(f"\nBenchmark Results:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Processed {max_codes} code snippets")
    print(f"Achieved QPS: {qps:.2f}")
    print(f"Avg processing time per code: {total_time * 1000 / max_codes:.2f} ms")

    return {
        "batch_size": batch_size,
        "max_workers": max_workers,
        "max_codes": max_codes,
        "total_time": total_time,
        "qps": qps
    }


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
        max_workers=2,  # g5.xlarge has 4 vCPUs
        max_codes=1000,  # Number of code snippets to process
    )

    # Optional: Try different configurations to find optimal settings
    # batch_sizes = [32, 64, 128]
    # for bs in batch_sizes:
    #     run_benchmark(batch_size=bs)
