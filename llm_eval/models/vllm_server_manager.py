"""
vLLM Server Management Module

This module provides functionality to start, manage, and shutdown vLLM servers
for local model inference. It handles server lifecycle, health checks, and
GPU memory cleanup.
"""

import atexit
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil
import requests
import torch
from huggingface_hub import HfApi

from ..wandb_singleton import WandbConfigSingleton


def start_vllm_server(vllm_config=None):
    """
    Start a vLLM server with configuration.
    
    Args:
        vllm_config (dict, optional): vLLM configuration. If None, uses WandbConfigSingleton.
    
    Returns:
        subprocess.Popen: The server process object
    """
    if vllm_config is not None:
        # Use provided config
        cfg = vllm_config
        run = None
    else:
        # Fallback to WandbConfigSingleton
        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config
        run = instance.run

    model_artifact_path = cfg.get("model", {}).get("artifacts_path", None)
    if model_artifact_path is not None and run is not None:
        artifact = run.use_artifact(model_artifact_path, type='model')
        artifact = Path(artifact.download())
        cfg["model"].update({"local_path": artifact / artifact.name.split(":")[0]})

    def run_vllm_server():
        model_id = cfg.get("pretrained_model_name_or_path", "")
        # Command to start the server
        command = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(model_id),
            "--dtype", cfg.get("dtype", "auto"),
            "--max-model-len", str(cfg.get("max_model_len", 4096)),
            "--max-num-seqs", str(cfg.get("batch_size", 8)),
            "--tensor-parallel-size", str(cfg.get("tensor_parallel_size", 1)),
            "--port", str(cfg.get("port", 8000)),
            "--seed", "42",
            "--uvicorn-log-level", "warning",
            "--disable-log-stats",
            "--disable-log-requests",
        ]

        # Add optional parameters
        if cfg.get("download_dir"):
            command.extend(["--download-dir", str(cfg.get("download_dir"))])
        if cfg.get("quantization"):
            command.extend(["--quantization", str(cfg.get("quantization"))])
        if cfg.get("revision"):
            command.extend(["--revision", str(cfg.get("revision"))])
        if cfg.get("trust_remote_code", False):
            command.append("--trust-remote-code")
        if cfg.get("reasoning_parser"):
            command.extend(["--reasoning-parser", str(cfg.get("reasoning_parser"))])

        # Run server in background using subprocess
        process = subprocess.Popen(command)

        # Save process ID to file
        with open('vllm_server.pid', 'w') as pid_file:
            pid_file.write(str(process.pid))
        # Wait for server to boot up
        time.sleep(10)

        # Return server process
        return process

    def health_check():
        """Check if the vLLM server is healthy and ready to accept requests."""
        port = cfg.get("port", 8000)
        url = f"http://localhost:{port}/health"
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print("Health check passed!")
                    break
                else:
                    print(f"Health check failed with status code: {response.status_code}")
            except requests.ConnectionError:
                print("Failed to connect to the server. Retrying...")
            time.sleep(10)  # Wait and retry

    # Start the server
    server_process = run_vllm_server()
    print("vLLM server is starting...")

    # Terminate the server when script ends
    def cleanup():
        print("Terminating vLLM server...")
        server_process.terminate()
        server_process.wait()

    atexit.register(cleanup)

    # Catch SIGTERM and gracefully terminate the server
    def handle_sigterm(sig, frame):
        print("SIGTERM received. Shutting down vLLM server gracefully...")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    # Wait until the server is fully launched
    health_check()
    
    return server_process


def force_cuda_memory_cleanup():
    """
    Force cleanup of CUDA memory to free up GPU resources.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        # Clear GPU memory for all CUDA devices
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def wait_for_gpu_memory_release(timeout=60):
    """
    Wait for GPU memory to be released after server shutdown.
    
    Args:
        timeout (int): Maximum time to wait in seconds
    """
    start_time = time.time()
    while torch.cuda.memory_allocated() > 0:
        if time.time() - start_time > timeout:
            print(f"Warning: GPU memory not fully released after {timeout} seconds.")
            break
        print(f"Waiting for GPU memory to be released. {torch.cuda.memory_allocated()} bytes still allocated.")
        force_cuda_memory_cleanup()
        time.sleep(1)


def shutdown_vllm_server():
    """
    Shutdown the vLLM server gracefully and clean up resources.
    """
    try:
        with open('vllm_server.pid', 'r') as pid_file:
            pid = int(pid_file.read().strip())
        
        process = psutil.Process(pid)
        
        # Terminate the process synchronously
        process.terminate()
        try:
            process.wait(timeout=30)
        except psutil.TimeoutExpired:
            print("Termination timed out. Killing the process.")
            process.kill()
        
        print(f"vLLM server with PID {pid} has been terminated.")
        
        # Remove PID file
        os.remove('vllm_server.pid')
        
    except psutil.NoSuchProcess:
        print(f"Process with PID {pid} not found. It may have already been terminated.")
    except FileNotFoundError:
        print("PID file not found. vLLM server may not be running.")
    except Exception as e:
        print(f"An error occurred while shutting down vLLM server: {e}")
    finally:
        # GPU memory cleanup
        print("Cleaning up GPU memory...")
        force_cuda_memory_cleanup()
        wait_for_gpu_memory_release()
        print("GPU memory cleanup completed.")

    # Wait a bit to ensure resources are freed
    time.sleep(10)
