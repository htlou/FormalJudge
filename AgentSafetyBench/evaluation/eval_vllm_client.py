"""
vLLM Client Evaluation Script.

This script connects to an existing vLLM server and runs evaluation.
It uses the same evaluation logic as eval_parallel.py but with VllmAPI.

Usage:
    python eval_vllm_client.py --server_url http://localhost:8000/v1 \
                               --model_name /path/to/model \
                               --num_workers 50

    # Or using connection info file:
    python eval_vllm_client.py --connection_file ./vllm_server_info/Qwen2.5-32B-Instruct_connection.json \
                               --num_workers 50
"""

import json
import sys
import time
import requests
from tqdm import tqdm
from copy import deepcopy
import os
import ray
sys.path.append('../environments')
from redirect_output import redirect_output
from EnvManager import EnvManager
from model_api import VllmAPI
from argparse import ArgumentParser


def wait_for_server(server_url, timeout=300):
    """Wait for vLLM server to be ready before starting evaluation."""
    # Convert /v1 URL to health endpoint
    base_url = server_url.rstrip('/').replace('/v1', '')
    health_url = f"{base_url}/health"

    print(f"Waiting for vLLM server at {base_url} to be ready...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print(f"vLLM server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass

        print(".", end="", flush=True)
        time.sleep(2)

    print(f"\nTimeout waiting for server after {timeout}s")
    return False


parser = ArgumentParser()
# Server connection options
parser.add_argument("--server_url", type=str, default=None,
                    help="vLLM server URL (e.g., http://localhost:8000/v1)")
parser.add_argument("--model_name", type=str, default=None,
                    help="Model name as served by vLLM (usually the full model path)")
parser.add_argument("--connection_file", type=str, default=None,
                    help="Path to connection info JSON file (alternative to --server_url)")

# Output options
parser.add_argument("--output_name", type=str, default=None,
                    help="Name for output directory (defaults to model basename)")

# Evaluation options
parser.add_argument('--greedy', type=int, default=0,
                    help="Use greedy decoding (1) or sampling (0)")
parser.add_argument("--regen_exceed", type=int, default=0,
                    help="Regenerate samples that exceeded max rounds")
parser.add_argument('--extra_info', type=str, default='',
                    help="Extra info to append to output directory name")
parser.add_argument('--allow_empty', type=int, default=0,
                    help="Allow empty responses")
parser.add_argument('--num_workers', type=int, default=50,
                    help="Number of Ray workers for concurrent API calls")

args = parser.parse_args()

# Load connection info
if args.connection_file:
    with open(args.connection_file, 'r') as f:
        conn_info = json.load(f)
    server_url = conn_info.get('url') or conn_info.get('local_url')
    model_name = conn_info.get('model_name')
    output_name = args.output_name or conn_info.get('model_short_name')
elif args.server_url and args.model_name:
    server_url = args.server_url
    model_name = args.model_name
    output_name = args.output_name or os.path.basename(model_name.rstrip('/'))
else:
    parser.error("Either --connection_file or both --server_url and --model_name are required")

num_workers = args.num_workers

# Generation config
if args.greedy:
    print("Use greedy decoding!")
    generation_config = {
        "temperature": 0.,
        "max_tokens": 8192
    }
else:
    print("Use random sampling!")
    generation_config = {
        "temperature": 1.0,
        "max_tokens": 8192
    }


@ray.remote
class EvaluatorWorker:
    """Ray worker class that calls vLLM server via API."""

    def __init__(self, server_url, served_model_name, generation_config, allow_empty):
        # Set up sys.path for worker process
        import sys
        import os
        worker_dir = os.path.dirname(os.path.abspath(__file__))
        environments_path = os.path.abspath(os.path.join(worker_dir, '../environments'))
        if environments_path not in sys.path:
            sys.path.insert(0, environments_path)

        # Import in worker process
        from EnvManager import EnvManager
        from model_api import VllmAPI

        # Use served_model_name (model_path) for API calls - vLLM uses the full path as model name
        self.agent_api = VllmAPI(
            model_name=served_model_name,
            generation_config=generation_config,
            base_url=server_url,
            api_key="EMPTY"
        )
        self.env_manager = EnvManager()
        self.allow_empty = allow_empty

    def parse_envs(self, envs_info):
        envs = []
        tool_descs = []
        for env_info in envs_info:
            env_name = env_info['name']
            if 'parameters' in env_info:
                env_params = env_info['parameters'] or None
            else:
                env_params = None
            env = self.env_manager.init_env(env_name, env_params)
            if env is None:
                print(envs_info)
                raise ValueError(f"Environment {env_name} not found.")
            envs.append(env)

            tool_names = env_info['tools']
            tool_desc = env.get_tool_descs(tool_names)
            tool_descs.extend(tool_desc)

        return envs, tool_descs

    def evaluate(self, d):
        envs = []
        tool_descs = []

        max_rounds = 10
        rounds = 0
        final_result = None

        info = {"id": d["id"]}
        messages = None
        try:
            messages = [{"role": "system", "content": self.agent_api.get_system_prompt(d)}]

            if "dialog" in d:
                messages.extend(d["dialog"])
            else:
                messages.append({"role": "user", "content": d["instruction"]})

            if d['environments'] and d['environments'][0]['name'] != '':
                envs, tool_descs = self.parse_envs(d['environments'])

                for item in tool_descs:
                    item['type'] = "object"

                for i, item in enumerate(tool_descs):
                    newitem = {'type': 'function', 'function': item}
                    tool_descs[i] = newitem

            while rounds < max_rounds:
                if rounds > 8:
                    print(f'current round: {rounds}, messages: {messages}')
                else:
                    print(f'id {d["id"]}, current round: {rounds}')

                completion = self.agent_api.generate_response(messages, tool_descs)

                rounds += 1

                if completion is None or 'type' not in completion:
                    continue
                if completion['type'] == 'error':
                    print(completion['message'])
                    continue

                if completion['type'] == 'tool':
                    tool_call_id = completion['tool_call_id']
                    tool_name = completion['tool_name']
                    arguments = completion['arguments']

                    tool_call_result = None
                    for env_idx, env in enumerate(envs):
                        provided_tools = d['environments'][env_idx]['tools']
                        if tool_name not in provided_tools:
                            continue
                        if not env.has_tool(tool_name):
                            continue
                        tool_call_result = env.call_tool(tool_name, deepcopy(arguments))
                        break

                    if tool_call_result is None:
                        tool_call_result = {"success": False, "message": f"Tool {tool_name} doesn't exist."}

                    tool_call_response = {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "arguments": json.dumps(arguments),
                                    "name": tool_name
                                }
                            }
                        ]
                    }

                    tool_call_result_message = {
                        "role": "tool",
                        "content": json.dumps(tool_call_result),
                        "tool_call_id": tool_call_id,
                        "name": tool_name
                    }

                    messages.append(tool_call_response)
                    messages.append(tool_call_result_message)

                # Normal content part
                else:
                    content = completion['content']
                    messages.append({
                        'role': 'assistant',
                        'content': content
                    })
                    final_result = messages
                    break

        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
            return {
                'result': None,
                'rounds': rounds,
                'error': True,
                'exceed': False,
                'messages': messages if messages else [],
                'info': info
            }

        if final_result is None:
            if rounds == 10:
                final_result = messages
                return {
                    'result': final_result,
                    'rounds': rounds,
                    'error': False,
                    'exceed': True,
                    'messages': messages,
                    'info': info
                }
            else:
                return {
                    'result': None,
                    'rounds': rounds,
                    'error': True,
                    'exceed': False,
                    'messages': messages if messages else [],
                    'info': info
                }
        else:
            if not self.allow_empty and (final_result[-1]['role'] != 'assistant' or final_result[-1]['content'] == ''):
                return {
                    'result': None,
                    'rounds': rounds,
                    'error': True,
                    'exceed': False,
                    'messages': messages,
                    'info': info
                }

            return {
                'result': final_result,
                'rounds': rounds,
                'error': False,
                'exceed': False,
                'messages': None,
                'info': info
            }


def remove_exceed_samples(outpath, exceed_log_path):
    try:
        with open(outpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        with open(exceed_log_path, 'r', encoding='utf-8') as f:
            exceed_samples = json.load(f)
    except Exception as e:
        return

    exceed_ids = set([d['info']['id'] for d in exceed_samples])
    new_data = []
    for d in data:
        if d['id'] in exceed_ids:
            continue
        new_data.append(d)

    print(f'remove {len(data) - len(new_data)} exceed samples')

    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=1, ensure_ascii=False)


def clear_log(error_log_path, exceed_log_path, clear_error=True, clear_exceed=True):
    if clear_error:
        try:
            with open(error_log_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
        except:
            pass
    if clear_exceed:
        try:
            with open(exceed_log_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
        except:
            pass


def eval_file(path, outpath, error_log_path, exceed_log_path, server_url, served_model_name, generation_config, allow_empty, num_workers):
    basedir = os.path.dirname(outpath)
    os.makedirs(basedir, exist_ok=True)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if os.path.exists(outpath):
        try:
            with open(outpath, 'r', encoding='utf-8') as f:
                outdata = json.load(f)
        except Exception as e:
            print(e)
            outdata = []
    else:
        outdata = []

    _outdata = []
    for d in outdata:
        if 'label' in d and d['label'] == -3:
            # changed samples should be re-evaluated
            continue
        _outdata.append(d)
    outdata = _outdata

    outids = set([d['id'] for d in outdata])
    _data = []
    print("Already evaluated: ", len(outids))
    for d in data:
        if d['id'] in outids:
            continue
        if 'finish' in d:
            if d['finish'] != 1:
                continue
        _data.append(d)

    data = _data

    if len(data) == 0:
        print("No new samples to evaluate.")
        return

    print(f"Evaluating {len(data)} samples with {num_workers} workers")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=num_workers, ignore_reinit_error=True)

    # Create worker pool - all workers connect to the same vLLM server
    workers = [
        EvaluatorWorker.remote(server_url, served_model_name, generation_config, allow_empty)
        for _ in range(num_workers)
    ]

    # Process data in parallel
    success_count = 0
    fail_count = 0
    exceed_count = 0
    error_samples = []
    exceed_samples = []

    # Use round-robin assignment to workers
    futures = []
    sample_indices = []

    for idx, d in enumerate(data):
        worker = workers[idx % len(workers)]
        future = worker.evaluate.remote(d)
        futures.append(future)
        sample_indices.append(idx)

    # Collect results with progress bar
    results = []
    for future, idx in tqdm(zip(futures, sample_indices), total=len(futures), desc="Evaluating"):
        try:
            result = ray.get(future, timeout=120)  # 2 minute timeout per sample
            results.append((idx, result))
        except ray.exceptions.GetTimeoutError:
            print(f"Timeout processing sample {data[idx]['id']}")
            results.append((idx, {
                'result': None,
                'rounds': 0,
                'error': True,
                'exceed': False,
                'messages': [],
                'info': {'id': data[idx]['id']}
            }))
        except Exception as e:
            print(f"Error processing sample {data[idx]['id']}: {e}")
            results.append((idx, {
                'result': None,
                'rounds': 0,
                'error': True,
                'exceed': False,
                'messages': [],
                'info': {'id': data[idx]['id']}
            }))

    # Sort results by original index
    results.sort(key=lambda x: x[0])

    # Process results and write output
    save_interval = 100  # Save every N samples
    for i, (idx, result) in enumerate(results):
        d = data[idx]
        if result['result'] is not None:
            success_count += 1
            if result['rounds'] == 10:
                print(f'Exceed max rounds! id: {d["id"]}')
                exceed_count += 1

            d['output'] = result['result']
            outdata.append(d)
        else:
            fail_count += 1
            print(f'Fail! id: {d["id"]}')

        # Collect error and exceed samples
        if result['error']:
            error_samples.append({'messages': result['messages'], 'info': result['info']})
        if result['exceed']:
            exceed_samples.append({'messages': result['messages'], 'info': result['info']})

        # Save intermediate results periodically
        if (i + 1) % save_interval == 0:
            with open(outpath, 'w', encoding='utf-8') as fw:
                json.dump(outdata, fw, indent=2, ensure_ascii=False)
            print(f"Saved intermediate results: {len(outdata)} samples")

    # Write output file
    with open(outpath, 'w', encoding='utf-8') as fw:
        json.dump(outdata, fw, indent=2, ensure_ascii=False)

    # Write error and exceed logs
    if error_samples:
        try:
            with open(error_log_path, 'r', encoding='utf-8') as f:
                existing_errors = json.load(f)
        except:
            existing_errors = []
        existing_errors.extend(error_samples)
        with open(error_log_path, 'w', encoding='utf-8') as f:
            json.dump(existing_errors, f, indent=2, ensure_ascii=False)

    if exceed_samples:
        try:
            with open(exceed_log_path, 'r', encoding='utf-8') as f:
                existing_exceed = json.load(f)
        except:
            existing_exceed = []
        existing_exceed.extend(exceed_samples)
        with open(exceed_log_path, 'w', encoding='utf-8') as f:
            json.dump(existing_exceed, f, indent=2, ensure_ascii=False)

    print(f'Success: {success_count} (exceed: {exceed_count}), Failed: {fail_count}')

    # Shutdown Ray
    ray.shutdown()


if __name__ == '__main__':
    path = "../data/released_data.json"
    outdir = f"./evaluation_results/tot-{output_name}"

    outdir += f'{args.extra_info}'
    outpath = f"{outdir}/gen_res.json"
    error_log_path = f"{outdir}/error_samples.json"
    exceed_log_path = f"{outdir}/exceed_round_samples.json"
    log_path = f"{outdir}/gen_log.txt"

    os.makedirs(outdir, exist_ok=True)

    redirect_output(log_path, 'w')

    print("=" * 60)
    print("vLLM Client Evaluation")
    print(f"Server URL: {server_url}")
    print(f"Model name: {model_name}")
    print(f"Output name: {output_name}")
    print(f"Num workers: {num_workers}")
    print("=" * 60)

    # Wait for server to be ready before starting evaluation
    if not wait_for_server(server_url, timeout=300):
        print("Error: vLLM server is not available. Exiting.")
        sys.exit(1)

    regenerate_exceed_samples = args.regen_exceed
    if regenerate_exceed_samples:
        remove_exceed_samples(outpath, exceed_log_path)
        clear_log(error_log_path, exceed_log_path, clear_error=True, clear_exceed=True)
    else:
        clear_log(error_log_path, exceed_log_path, clear_error=True, clear_exceed=False)

    eval_file(path, outpath, error_log_path, exceed_log_path, server_url, model_name,
              generation_config, args.allow_empty, num_workers)
