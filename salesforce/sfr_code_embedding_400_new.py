import math
import time
import hashlib
from collections import defaultdict

import pandas as pd
import torch
import concurrent.futures
from transformers import AutoModel, AutoTokenizer
from typing import List, Union, Dict

from constants.constants import ModelCheckPoints

# Define constants
MAX_CHARACTER_SIZE = 8192


# Define ModelCheckPoints enum-like class if not imported from constants


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class SfrCodeEmbedding400:
    def __init__(self, checkpoint: str, max_length: int = MAX_CHARACTER_SIZE, enable_cache: bool = True):
        print(f"Loading model from checkpoint: {checkpoint}")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, add_eos_token=True)
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
        self.enable_cache = enable_cache
        self.embedding_cache = {} if enable_cache else None

        # Optimize model for inference
        self.model.eval()
        if device == "cuda":
            self.model = self.model.half()  # Use FP16 for faster inference on GPU
            torch.backends.cudnn.benchmark = True  # Optimize CUDA operations

            # Try to use BetterTransformer if available
            try:
                from transformers.utils import is_flash_attn_available
                if is_flash_attn_available():
                    print("Flash Attention available - using BetterTransformer")
                    self.model = self.model.to_bettertransformer()
            except (ImportError, AttributeError):
                print("BetterTransformer not available - using standard model")

            # Try to compile model if PyTorch 2.0+ is available
            if hasattr(torch, 'compile'):
                try:
                    print("Compiling model with torch.compile()...")
                    self.model = torch.compile(self.model)
                except Exception as e:
                    print(f"Could not compile model: {e}")

        # Clear cache after initialization
        if device == "cuda":
            torch.cuda.empty_cache()

    def _hash_input(self, input_text: str) -> str:
        """Create a hash key for caching."""
        return hashlib.md5(input_text.encode()).hexdigest()

    def create_embedding(self, inputs: Union[str, List[str]], use_cache: bool = True) -> torch.Tensor:
        """Create embeddings for a single input or a list of inputs"""
        if isinstance(inputs, str):
            inputs = [inputs]

        # Check cache if enabled
        if self.enable_cache and use_cache:
            results = []
            uncached_inputs = []
            uncached_indices = []

            for i, input_text in enumerate(inputs):
                cache_key = self._hash_input(input_text)
                if cache_key in self.embedding_cache:
                    results.append(self.embedding_cache[cache_key])
                else:
                    uncached_inputs.append(input_text)
                    uncached_indices.append((i, cache_key))

            if not uncached_inputs:
                # All inputs were cached
                return torch.cat(results, dim=0) if len(results) > 1 else results[0]

            # Process only uncached inputs
            inputs = uncached_inputs

        # Compute dynamic max length based on inputs to avoid unnecessary padding
        max_input_length = min(self.max_length, max(len(input_text) for input_text in inputs) + 50)

        # Use torch.amp.autocast with explicit device arg to avoid FutureWarning
        if device == "cuda":
            with torch.amp.autocast(device_type='cuda'):  # Using updated API
                tokens = self.tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_input_length
                ).to(device)

                with torch.no_grad():  # Disable gradients for inference
                    embeddings = self.model(**tokens).last_hidden_state
        else:
            # CPU version without autocast
            tokens = self.tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length
            ).to(device)

            with torch.no_grad():  # Disable gradients for inference
                embeddings = self.model(**tokens).last_hidden_state

        # Cache results if enabled
        if self.enable_cache and use_cache and 'uncached_indices' in locals():
            for (idx, cache_key), embedding in zip(uncached_indices, embeddings):
                self.embedding_cache[cache_key] = embedding
                results.insert(idx, embedding)
            return torch.cat(results, dim=0) if len(results) > 1 else results[0]

        return embeddings

    def create_embedding_with_streams(self, inputs: List[str], num_streams: int = 2) -> torch.Tensor:
        """Create embeddings using multiple CUDA streams for parallelization"""
        if not torch.cuda.is_available() or len(inputs) < num_streams or num_streams < 2:
            # Fall back to standard embedding for non-CUDA or small batches
            return self.create_embedding(inputs)

        # For larger batches, catch any OOM errors and fall back to standard processing
        try:
            # Split inputs into chunks for each stream
            chunk_size = max(1, len(inputs) // num_streams)
            chunks = [inputs[i:i + chunk_size] for i in range(0, len(inputs), chunk_size)]
            chunks = chunks[:num_streams]  # Limit to num_streams chunks

            results = []
            streams = [torch.cuda.Stream() for _ in range(len(chunks))]

            for chunk, stream in zip(chunks, streams):
                with torch.cuda.stream(stream):
                    result = self.create_embedding(chunk, use_cache=False)
                    results.append(result)

            # Synchronize all streams
            torch.cuda.synchronize()

            return torch.cat(results, dim=0) if len(results) > 1 else results[0]
        except RuntimeError as e:
            # If we get an OOM error, clear cache and fall back
            print(f"Stream processing error: {e}. Falling back to standard processing.")
            torch.cuda.empty_cache()
            return self.create_embedding(inputs)

    def process_batches(self, code_snippets: List[str], batch_size: int = 64) -> List[torch.Tensor]:
        """Process code snippets in optimized batches"""
        results = []
        num_batches = len(code_snippets) // batch_size + (1 if len(code_snippets) % batch_size > 0 else 0)

        for i in range(num_batches):
            batch = code_snippets[i * batch_size:(i + 1) * batch_size]

            try:
                # Use streams for larger batches on CUDA if possible
                if device == "cuda" and len(batch) >= 8:
                    embeddings = self.create_embedding_with_streams(batch)
                else:
                    embeddings = self.create_embedding(batch)

                results.append(embeddings)
                print(f"Processed batch {i + 1}/{num_batches}")

            except RuntimeError as e:
                # Handle OOM errors by processing in smaller batches
                print(f"Error processing batch: {e}")
                print("Trying with smaller batch size...")

                # Reduce batch size and try again
                smaller_batch_size = max(1, len(batch) // 2)
                for j in range(0, len(batch), smaller_batch_size):
                    smaller_batch = batch[j:j + smaller_batch_size]
                    try:
                        embeddings = self.create_embedding(smaller_batch)
                        results.append(embeddings)
                    except RuntimeError:
                        print(f"Still unable to process batch - skipping {len(smaller_batch)} items")

            # Clear cache between batches
            if i % 2 == 0 and device == "cuda":
                torch.cuda.empty_cache()

        return results

    def dynamic_batching(self, code_snippets: List[str]) -> List[torch.Tensor]:
        """Group snippets by length and use appropriate batch sizes"""
        # Group snippets by approximate length
        groups = defaultdict(list)

        for snippet in code_snippets:
            length = len(snippet)
            if length < 1000:
                groups['short'].append(snippet)
            elif length < 4000:
                groups['medium'].append(snippet)
            else:
                groups['long'].append(snippet)

        # Process each group with appropriate batch size
        results = []
        batch_sizes = {'short': 128, 'medium': 64, 'long': 16}  # Reduced long batch size

        for group_name, snippets in groups.items():
            if snippets:
                print(f"Processing {len(snippets)} {group_name} snippets with batch size {batch_sizes[group_name]}")
                results.extend(self.process_batches(snippets, batch_size=batch_sizes[group_name]))

        return results

    def process_in_parallel(self, code_snippets: List[str], batch_size: int = 10, max_workers: int = None) -> List[
        torch.Tensor]:
        """Process snippets using parallel execution"""
        # Set sensible default max_workers
        if max_workers is None:
            max_workers = 1 if device == "cuda" else min(4, math.floor(torch.get_num_threads() * 0.5))

        print(f"Processing with {max_workers} workers")

        # For CUDA, use dynamic batching with streams instead of multiple workers
        if device == "cuda":
            return self.dynamic_batching(code_snippets)

        # For CPU, use ThreadPoolExecutor
        results = []
        num_batches = len(code_snippets) // batch_size + (1 if len(code_snippets) % batch_size > 0 else 0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(num_batches):
                batch = code_snippets[i * batch_size:(i + 1) * batch_size]
                futures.append(executor.submit(self.create_embedding, batch))

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    results.append(future.result())
                    print(f"Completed batch {i + 1}/{num_batches}")
                except Exception as e:
                    print(f"Error processing batch: {e}")

        return results


def get_code_search_code_snippets():
    """Load code snippets from CSV file"""
    try:
        code_search_net_dataset_path = "benchmarks/csn_10k.csv"
        records = pd.read_csv(code_search_net_dataset_path)
        code_snippets = records['code'].tolist()
        return code_snippets
    except Exception as e:
        print(f"Error loading CSV: {e}")
        # Return dummy data if file not found
        return ["def hello(): print('hello world')"] * 100


def get_test_code_records(count: int, static: bool = False) -> List[str]:
    """Get test code snippets for benchmarking"""
    if static:
        # Example input code - making it shorter to avoid memory issues
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
        code_snippets = get_code_search_code_snippets()[:count]

    return code_snippets


def run_benchmark(
        batch_size: int = 64,
        max_workers: int = 4,
        max_codes: int = 1000,
        max_length: int = MAX_CHARACTER_SIZE,
        enable_cache: bool = True
) -> Dict:
    """Run a benchmark to measure QPS with different configurations"""
    print(f"Starting benchmark with batch_size={batch_size}, max_workers={max_workers}, max_codes={max_codes}")

    # Initialize model
    sfr_embedder = SfrCodeEmbedding400(
        ModelCheckPoints.SFR_SMALL.value,
        max_length=max_length,
        enable_cache=enable_cache
    )

    # Get test code snippets
    code_snippets = get_test_code_records(max_codes, static=True)

    # Test with different input sizes to compare QPS differences
    # Use more reasonable sizes to avoid OOM errors
    small_len = min(500, len(code_snippets[0]) // 2)
    large_multiplier = min(5, 8192 // len(code_snippets[0]))  # Avoid making inputs too large

    large_snippets = [code_snippets[0] * large_multiplier] * (max_codes // large_multiplier)  # Larger texts
    small_snippets = [code_snippets[0][:small_len]] * max_codes  # Smaller texts

    # GPU warmup
    print("Warming up GPU...")
    _ = sfr_embedder.create_embedding(code_snippets[:min(5, len(code_snippets))])
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Function to safely run a benchmark
    def safe_benchmark(snippets, bs, name):
        try:
            print(f"\nRunning benchmark with {name} inputs...")
            start_time = time.time()
            _ = sfr_embedder.process_in_parallel(snippets, batch_size=bs, max_workers=max_workers)
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()

            total_time = end_time - start_time
            qps = len(snippets) / total_time

            print(f"{name} inputs: {qps:.2f} QPS (avg time: {total_time * 1000 / len(snippets):.2f} ms)")

            # Clear cache
            if device == "cuda":
                torch.cuda.empty_cache()

            return total_time, qps
        except Exception as e:
            print(f"Error during {name} benchmark: {e}")
            return None, 0

    # Run benchmarks with different input sizes
    _, regular_qps = safe_benchmark(code_snippets, batch_size, "regular")
    _, small_qps = safe_benchmark(small_snippets, batch_size * 2, "small")
    _, large_qps = safe_benchmark(large_snippets, max(1, batch_size // 2), "large")

    # Print summary
    print(f"\nBenchmark Summary:")
    print(f"Regular inputs: {regular_qps:.2f} QPS")
    print(f"Small inputs: {small_qps:.2f} QPS")
    print(f"Large inputs: {large_qps:.2f} QPS")

    # Print recommendations
    if large_qps > 0:
        print("\nRecommendations to improve QPS:")

        if large_qps < small_qps / 2:
            print("- For large inputs: Consider chunking large inputs and averaging embeddings")
            print("- Adjust batch size dynamically based on input length")

        print("- Experiment with different batch sizes (try: 16, 32, 64, 128)")
        print("- Consider model quantization for higher throughput")
        print("- If possible, use a GPU with more memory")

    return {
        "batch_size": batch_size,
        "max_workers": max_workers,
        "max_codes": max_codes,
        "enable_cache": enable_cache,
        "qps": regular_qps,
        "small_qps": small_qps,
        "large_qps": large_qps
    }


def benchmark_configurations() -> Dict[str, float]:
    """Test different configurations and return the best one"""
    # Define configurations to test
    configs = [
        {"batch_size": 32, "max_workers": 1, "enable_cache": True},
        {"batch_size": 64, "max_workers": 1, "enable_cache": True},
        {"batch_size": 16, "max_workers": 1, "enable_cache": True},
        {"batch_size": 32, "max_workers": 2, "enable_cache": True},
    ]

    results = {}

    for config in configs:
        print(f"\nTesting configuration: {config}")
        bs = config["batch_size"]
        workers = config["max_workers"]
        cache = config["enable_cache"]

        # Run benchmark with this configuration
        try:
            result = run_benchmark(
                batch_size=bs,
                max_workers=workers,
                max_codes=200,  # Use fewer codes for quick testing
                enable_cache=cache
            )

            config_key = f"bs{bs}_w{workers}_c{int(cache)}"
            results[config_key] = result["qps"]
        except Exception as e:
            print(f"Error testing configuration {config}: {e}")

    if results:
        # Find best configuration
        best_config = max(results.items(), key=lambda x: x[1])
        print(f"\nBest configuration: {best_config[0]} with QPS: {best_config[1]:.2f}")
    else:
        print("No successful configuration tests")

    return results


if __name__ == "__main__":
    # Set CUDA device options for best performance
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
        torch.backends.cudnn.allow_tf32 = True

        # Set memory allocation strategy more conservatively
        try:
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
        except:
            print("Could not set memory fraction - continuing without it")

    # Print system info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        try:
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        except:
            print("Could not get GPU memory info")

    # Choose benchmark mode
    benchmark_mode = "single"  # Options: "single", "configurations"

    if benchmark_mode == "configurations":
        # Test multiple configurations to find optimal settings
        benchmark_configurations()
    else:
        # Run a single benchmark with default settings
        results = run_benchmark(
            batch_size=32,  # Using a more conservative batch size
            max_workers=2,  # g5.xlarge has 4 vCPUs, use 2 for best results
            max_codes=500,  # Reduced for stability
            enable_cache=True  # Enable embedding cache for repeated inputs
        )