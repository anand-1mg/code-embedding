import time
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Union

from constants.constants import ModelCheckPoints

# Define constants
MAX_CHARACTER_SIZE = 8192


# Define ModelCheckPoints enum-like class if not imported from constants


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


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

        # Clear cache after initialization
        if device == "cuda":
            torch.cuda.empty_cache()

    def create_embedding(self, inputs: Union[str, List[str]]) -> torch.Tensor:
        """Create embeddings for a single input or a list of inputs - optimized for speed"""
        if isinstance(inputs, str):
            inputs = [inputs]

        # Simple length-based dynamic batch sizing
        avg_length = sum(len(text) for text in inputs) / len(inputs)

        # Compute tokenization with minimal overhead
        tokens = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(device)

        # Fast inference
        with torch.no_grad():
            if device == "cuda":
                with torch.cuda.amp.autocast():
                    embeddings = self.model(**tokens).last_hidden_state
            else:
                embeddings = self.model(**tokens).last_hidden_state

        return embeddings

    def process_batches(self, code_snippets: List[str], batch_size: int = 64) -> List[torch.Tensor]:
        """Process code snippets in optimized batches"""
        results = []

        # Dynamically adjust batch size based on average input length
        avg_length = sum(len(snippet) for snippet in code_snippets) / max(1, len(code_snippets))

        # Dynamic batch sizing based on input length
        if avg_length > 5000:
            batch_size = max(8, batch_size // 8)
        elif avg_length > 2000:
            batch_size = max(16, batch_size // 4)
        elif avg_length > 1000:
            batch_size = max(32, batch_size // 2)
        elif avg_length < 500:
            batch_size = min(256, batch_size * 2)

        print(f"Using dynamic batch size: {batch_size} for avg length: {avg_length:.0f}")

        num_batches = len(code_snippets) // batch_size + (1 if len(code_snippets) % batch_size > 0 else 0)

        for i in range(num_batches):
            batch = code_snippets[i * batch_size:(i + 1) * batch_size]
            embeddings = self.create_embedding(batch)
            results.append(embeddings)
            print(f"Processed batch {i + 1}/{num_batches}")

            # Clear cache periodically
            if i % 5 == 0 and device == "cuda":
                torch.cuda.empty_cache()

        return results


def get_test_code_records(count: int, static: bool = False) -> List[str]:
    """Get test code snippets for benchmarking"""
    if static:
        # Example input code
        code = """
        class PRDiffHandler:
        def __init__(self, pr_service):
            self.pr_service = pr_service
            self.pr_diff = None
            self.pr_diff_mappings = {}
            self.pr_diffs = []

        async def get_effective_pr_diff(self, operation="code_review", agent_id=None):
            if not self.pr_diff:
                self.pr_diff = await self.pr_service.get_commit_diff_or_pr_diff()
            if not self.pr_diff_mappings:
                self.set_pr_diff_mappings(operation)

            if operation == "chat":
                diff_index = self.pr_diff_mappings.get("chat")
            else:
                if agent_id:
                    diff_index = self.pr_diff_mappings[agent_id]
                else:
                    diff_index = self.pr_diff_mappings["global_diff"]

            return self.pr_diffs[diff_index]
        """
        code_snippets = [code] * count
    else:
        try:
            import pandas as pd
            code_search_net_dataset_path = "benchmarks/csn_10k.csv"
            records = pd.read_csv(code_search_net_dataset_path)
            code_snippets = records['code'].tolist()[:count]
        except Exception as e:
            print(f"Error loading CSV: {e}")
            # Return dummy data if file not found
            code_snippets = ["def hello(): print('hello world')"] * count

    return code_snippets


def run_benchmark(batch_size: int = 64, max_codes: int = 1000):
    """Run a benchmark to measure QPS with different configurations - simplified for speed"""
    print(f"Starting benchmark with batch_size={batch_size}, max_codes={max_codes}")

    # Initialize model
    sfr_embedder = SfrCodeEmbedding400(ModelCheckPoints.SFR_SMALL.value)

    # Get test code snippets
    code_snippets = get_test_code_records(max_codes, static=True)

    # Create modified snippets for testing
    small_snippets = [code_snippets[0][:500]] * max_codes  # Smaller texts
    large_snippets = [code_snippets[0] * 5] * (max_codes // 5)  # Larger texts

    # GPU warmup
    print("Warming up GPU...")
    _ = sfr_embedder.create_embedding(code_snippets[:min(5, len(code_snippets))])
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Benchmark with regular inputs
    print("\nBenchmarking with regular inputs...")
    start_time = time.time()
    _ = sfr_embedder.process_batches(code_snippets, batch_size=batch_size)
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    regular_time = end_time - start_time
    regular_qps = max_codes / regular_time
    print(f"Regular inputs: {regular_qps:.2f} QPS (avg time: {regular_time * 1000 / max_codes:.2f} ms)")

    # Clear cache
    if device == "cuda":
        torch.cuda.empty_cache()

    # Benchmark with small inputs
    print("\nBenchmarking with small inputs...")
    start_time = time.time()
    _ = sfr_embedder.process_batches(small_snippets, batch_size=batch_size * 2)
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    small_time = end_time - start_time
    small_qps = max_codes / small_time
    print(f"Small inputs: {small_qps:.2f} QPS (avg time: {small_time * 1000 / max_codes:.2f} ms)")

    # Clear cache
    if device == "cuda":
        torch.cuda.empty_cache()

    # Benchmark with large inputs
    print("\nBenchmarking with large inputs...")
    start_time = time.time()
    _ = sfr_embedder.process_batches(large_snippets, batch_size=batch_size // 2)
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    large_time = end_time - start_time
    large_qps = (max_codes // 5) / large_time
    print(f"Large inputs: {large_qps:.2f} QPS (avg time: {large_time * 1000 / (max_codes // 5):.2f} ms)")

    # Print summary
    print("\nBenchmark Summary:")
    print(f"Regular inputs: {regular_qps:.2f} QPS")
    print(f"Small inputs: {small_qps:.2f} QPS")
    print(f"Large inputs: {large_qps:.2f} QPS")

    return {
        "regular_qps": regular_qps,
        "small_qps": small_qps,
        "large_qps": large_qps
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

    # First run with default batch size
    print("\n=== RUNNING WITH DEFAULT BATCH SIZE ===")
    results_default = run_benchmark(batch_size=64, max_codes=1000)

    # Second run with increased batch size for small inputs
    print("\n=== RUNNING WITH OPTIMIZED BATCH SIZE FOR SMALL INPUTS ===")
    results_optimized = run_benchmark(batch_size=128, max_codes=1000)