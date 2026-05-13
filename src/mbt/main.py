import argparse
import importlib
import json
import pkgutil
from pathlib import Path

from mbt.registry import API_REGISTRY, TASK_REGISTRY


def recursive_import(package_name):
    package = importlib.import_module(package_name)
    if hasattr(package, "__path__"):
        for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            importlib.import_module(name)


def main():
    """
    Main entry point for the MBT pipeline.

    This function orchestrates the end-to-end execution of a task by:
    1. Parsing command-line arguments for task, API, and script configurations.
    2. Setting up the environment (loading .env, disabling caching).
    3. Dynamically discovering and registering all available `Task` and `API` implementations.
    4. Executing the workflow:
      - `task.preprocess`: Prepares data and requests.
      - `api.process`: Performs inference using the specified backend (e.g., vLLM, OpenAI).
      - `task.postprocess`: Handles result formatting, metric calculation, or storage.
    """
    parser = argparse.ArgumentParser(description="Main entry point for the MBT pipeline.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--task-name", type=str, required=True, help='The registered name of the task to run (e.g., "musique", "math500").')
    parser.add_argument("--task-config", type=str, default="{}", help="JSON string containing configuration for the task.")
    parser.add_argument("--api-name", type=str, default=None, help='The registered name of the API backend (e.g., "vllm.chat").')
    parser.add_argument("--api-config", type=str, default="{}", help="JSON string containing configuration for the API.")
    parser.add_argument("--script-config", type=str, default="{}", help='JSON string for global script settings (e.g., "root_dir", "load_dotenv").')
    args = parser.parse_args()

    args.task_config = json.loads(args.task_config)
    args.api_config = json.loads(args.api_config)
    args.script_config = json.loads(args.script_config)

    if args.script_config.get("load_dotenv", True):
        from dotenv import load_dotenv

        load_dotenv()

    if args.script_config.get("hf_disable_caching", True):
        from datasets import disable_caching

        disable_caching()

    packages = ["mbt.apis", "mbt.tasks"]
    for pkg in packages:
        recursive_import(pkg)

    task = TASK_REGISTRY[args.task_name](args.task_config)
    task_dir = task.preprocess(Path(args.script_config.get("root_dir", "output")))
    if task_dir is not None:
        api = API_REGISTRY[args.api_name](args.api_config)
        api_dir = api.process(task_dir)
        task.postprocess(api_dir)


if __name__ == "__main__":
    main()
