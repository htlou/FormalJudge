"""
vLLM Server Starter.

This script starts a vLLM OpenAI-compatible server and returns connection info.
The server runs as a subprocess and can be managed via the returned process handle.

Usage:
    # As a module
    python vllm_server.py --model_path /path/to/model --port 8000

    # Programmatically
    from vllm_server import VllmServer
    server = VllmServer(model_path, port=8000)
    info = server.start()
    print(info)  # {'host': '0.0.0.0', 'port': 8000, 'url': 'http://...', 'model_name': '...'}
    server.stop()
"""

import subprocess
import time
import signal
import os
import sys
import socket
import requests
import json
import atexit
from argparse import ArgumentParser


def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        # Create a socket to determine the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


def find_free_port(start_port=8000, max_tries=100):
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_tries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", port))
            s.close()
            return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find a free port in range {start_port}-{start_port + max_tries}")


class VllmServer:
    """Manages a vLLM OpenAI-compatible server."""

    def __init__(
        self,
        model_path,
        host="0.0.0.0",
        port=8888,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=32768,
        tool_call_parser="hermes",
        auto_find_port=True,
    ):
        """
        Initialize vLLM server configuration.

        Args:
            model_path: Path to the model or HuggingFace model name
            host: Host to bind the server (default: 0.0.0.0 for all interfaces)
            port: Port number (default: 8000)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            max_model_len: Maximum model context length
            tool_call_parser: Tool call parser (hermes, llama3_json, mistral, etc.)
            auto_find_port: If True, find a free port if the specified one is in use
        """
        self.model_path = model_path
        self.model_name = os.path.basename(model_path.rstrip('/'))
        self.host = host
        self.port = port
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tool_call_parser = tool_call_parser
        self.auto_find_port = auto_find_port
        self.process = None
        self.local_ip = get_local_ip()

    def start(self, timeout=300, log_file=None):
        """
        Start the vLLM server.

        Args:
            timeout: Maximum seconds to wait for server to be ready (default: 300)
            log_file: Optional file path to redirect server output

        Returns:
            dict: Connection info with keys:
                - host: Server host address
                - port: Server port
                - url: Full API URL (http://host:port/v1)
                - local_url: URL using localhost
                - internal_url: URL using internal IP
                - model_name: Name of the served model (full path for vLLM)
                - model_short_name: Short model name (basename)
        """
        if self.process is not None:
            raise RuntimeError("Server is already running")

        # Find a free port if needed
        if self.auto_find_port:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("", self.port))
                s.close()
            except OSError:
                print(f"Port {self.port} is in use, finding a free port...")
                self.port = find_free_port(self.port)
                print(f"Using port {self.port}")

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            "--port", str(self.port),
            "--host", self.host,
            "--trust-remote-code",
            "--disable-log-requests",
            "--enable-auto-tool-choice",
            "--tool-call-parser", self.tool_call_parser,
        ]

        print(f"Starting vLLM server...")
        print(f"Command: {' '.join(cmd)}")

        # Set up output redirection
        if log_file:
            log_fh = open(log_file, 'w')
            stdout = log_fh
            stderr = subprocess.STDOUT
        else:
            stdout = subprocess.PIPE
            stderr = subprocess.STDOUT

        # Start server process
        self.process = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            preexec_fn=os.setsid  # Create new process group for clean shutdown
        )

        # Register cleanup
        atexit.register(self.stop)

        # Wait for server to be ready
        print(f"Waiting for vLLM server to start (timeout: {timeout}s)...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
                if response.status_code == 200:
                    info = self.get_connection_info()
                    print(f"\nvLLM server is ready!")
                    print(f"  Local URL:    {info['local_url']}")
                    print(f"  Internal URL: {info['internal_url']}")
                    print(f"  Model name:   {info['model_name']}")
                    return info
            except requests.exceptions.RequestException:
                pass

            # Check if process died
            if self.process.poll() is not None:
                if not log_file:
                    stdout_data, _ = self.process.communicate()
                    print(f"vLLM server failed to start. Output:\n{stdout_data.decode()}")
                else:
                    print(f"vLLM server failed to start. Check log: {log_file}")
                self.process = None
                raise RuntimeError("vLLM server failed to start")

            time.sleep(2)
            print(".", end="", flush=True)

        print(f"\nTimeout waiting for vLLM server to start")
        self.stop()
        raise RuntimeError("Timeout waiting for vLLM server to start")

    def stop(self):
        """Stop the vLLM server."""
        if self.process is None:
            return

        print("Stopping vLLM server...")

        # Check if process is still running
        if self.process.poll() is not None:
            print("vLLM server already stopped")
            self.process = None
            return

        try:
            # Kill the entire process group
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Server didn't stop gracefully, force killing...")
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=5)
        except ProcessLookupError:
            print("Process already terminated")
        except Exception as e:
            print(f"Error stopping vLLM server: {e}")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except:
                pass

        self.process = None
        print("vLLM server stopped")

    def is_running(self):
        """Check if the server process is running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def is_healthy(self):
        """Check if the server is healthy and responding."""
        try:
            response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_connection_info(self):
        """Get connection information for the server."""
        return {
            "host": self.host,
            "port": self.port,
            "local_url": f"http://localhost:{self.port}/v1",
            "internal_url": f"http://{self.local_ip}:{self.port}/v1",
            "url": f"http://{self.local_ip}:{self.port}/v1",  # Default to internal
            "model_name": self.model_path,  # vLLM uses full path as model name
            "model_short_name": self.model_name,
        }

    def save_connection_info(self, filepath):
        """Save connection info to a JSON file."""
        info = self.get_connection_info()
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Connection info saved to: {filepath}")
        return info


def main():
    parser = ArgumentParser(description="Start a vLLM OpenAI-compatible server")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model or HuggingFace model name")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind the server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8888,
                        help="Port for vLLM server (default: 8888)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                        help="Fraction of GPU memory to use (0.0-1.0)")
    parser.add_argument("--max_model_len", type=int, default=32768,
                        help="Maximum model context length")
    parser.add_argument("--tool_call_parser", type=str, default="hermes",
                        help="Tool call parser (hermes, llama3_json, mistral, etc.)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="File to save connection info (JSON)")
    parser.add_argument("--log_file", type=str, default=None,
                        help="File to save server logs")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for Ctrl+C instead of exiting immediately")

    args = parser.parse_args()

    server = VllmServer(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tool_call_parser=args.tool_call_parser,
    )

    try:
        info = server.start(log_file=args.log_file)

        # Save connection info if requested
        if args.output_file:
            server.save_connection_info(args.output_file)
        else:
            # Print to stdout for scripts to capture
            print("\n--- Connection Info (JSON) ---")
            print(json.dumps(info, indent=2))
            print("--- End Connection Info ---\n")

        if args.wait:
            print("Server is running. Press Ctrl+C to stop.")
            try:
                while server.is_running():
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nReceived interrupt signal")
        else:
            print("Server started successfully. Use the connection info above to connect.")
            print("To keep the server running, use --wait flag.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
