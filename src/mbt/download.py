import argparse

from huggingface_hub import snapshot_download
from tenacity import retry, stop_after_attempt, wait_fixed

MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

DATASETS = [
    "dgslibisey/MuSiQue",
    "framolfese/2WikiMultihopQA",
    "hotpotqa/hotpot_qa",
]


def log_before_retry(retry_state):
    print(f"    - Attempt {retry_state.attempt_number} failed with: {retry_state.outcome.exception()}. Retrying...")


def suppress_failure(retry_state):
    print(f"    - All {retry_state.retry_object.stop.max_attempt_number} attempts failed for {retry_state.args[0]}. Skipping.")
    return None


def download_repository(repo_id, repo_type, max_retries, retry_delay, **kwargs):
    @retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(retry_delay), before_sleep=log_before_retry, retry_error_callback=suppress_failure)
    def _download_with_retry(repo_id):
        snapshot_download(repo_id=repo_id, repo_type=repo_type, **kwargs)

    _download_with_retry(repo_id)


def main():
    parser = argparse.ArgumentParser(description="Download models and datasets from Hugging Face Hub sequentially.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--max-retries", type=int, default=10, help="Maximum number of retry attempts for a failed download.")
    parser.add_argument("--retry-delay", type=int, default=60, help="Wait time in seconds between retries.")
    args = parser.parse_args()

    n_models = len(MODELS)
    print(f"--- Starting Model Downloads ({n_models} items) ---")
    for i, model in enumerate(MODELS, start=1):
        print(f"\n[{i:02d}/{n_models}] 🚀 Downloading model: {model}")
        download_repository(repo_id=model, repo_type="model", max_retries=args.max_retries, retry_delay=args.retry_delay, allow_patterns="*.safetensors")
        print(f"[{i:02d}/{n_models}] ✅ Finished model: {model}")

    n_datasets = len(DATASETS)
    if n_datasets > 0:
        print(f"\n--- Starting Dataset Downloads ({n_datasets} items) ---")
        for i, dataset in enumerate(DATASETS, start=1):
            print(f"\n[{i:02d}/{n_datasets}] 🚀 Downloading dataset: {dataset}")
            download_repository(repo_id=dataset, repo_type="dataset", max_retries=args.max_retries, retry_delay=args.retry_delay)
            print(f"[{i:02d}/{n_datasets}] ✅ Finished dataset: {dataset}")

    print("\n🎉 All download tasks are complete.")


if __name__ == "__main__":
    main()
