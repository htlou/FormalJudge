import json
import sys
from tqdm import tqdm
from copy import deepcopy
import os
import ray
from threading import Lock
import copy
sys.path.append('../environments')
from redirect_output import redirect_output
from EnvManager import EnvManager
from model_api import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="qwen2.5-14b")
parser.add_argument('--greedy', type=int, default=1)
parser.add_argument("--regen_exceed", type=int, default=0)
parser.add_argument('--extra_info', type=str, default='')
parser.add_argument('--allow_empty', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=4)


args = parser.parse_args()

# stop_token = '[FINISHED]'

model_name = args.model_name
num_workers = args.num_workers

if args.greedy:
    print("Use greedy decoding!")
    api_model_generation_config = {
        "temperature": 0.,
        "max_tokens": 8192
    }
    local_model_generation_config = {
        "do_sample": False,
        "repetition_penalty": 1.0,
        "num_beams": 1,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "use_cache": True,
        "max_new_tokens": 2048,
    }
else:
    print("Use random sampling!")
    api_model_generation_config = {
        "temperature": 1.0,
        "max_tokens": 8192
    }
    local_model_generation_config = {
        "do_sample": True,
        # "repetition_penalty": 1.0,
        "num_beams": 1,
        # "temperature": 1.0,
        # "top_k": 0,
        # "top_p": 1.0,
        "use_cache": True,
        "max_new_tokens": 2048,
    }

def create_agent_api(model_name, api_model_generation_config, local_model_generation_config):
    """Create and return an agent API instance based on model name."""
    if model_name == 'gpt4omini':
        return OpenaiAPI(model_name='gpt-4o-mini', generation_config=api_model_generation_config)
    elif model_name == 'gpt4o':
        return OpenaiAPI(model_name='gpt-4o', generation_config=api_model_generation_config)
    elif model_name == 'gpt4-turbo':
        return OpenaiAPI(model_name='gpt-4-turbo', generation_config=api_model_generation_config)
    elif model_name == 'gpt-5':
        return OpenaiAPI(model_name='gpt-5', generation_config=api_model_generation_config)
    elif "gpt" in model_name:
        return OpenaiAPI(model_name=model_name, generation_config=api_model_generation_config)
    elif model_name == 'claude3-opus':
        return ClaudeAPI(model_name='anthropic/claude-3-opus', generation_config=api_model_generation_config)
    elif model_name == 'claude3.5-sonnet':
        return ClaudeAPI(model_name='anthropic/claude-3.5-sonnet', generation_config=api_model_generation_config)
    elif model_name == 'claude3.5-haiku':
        return ClaudeAPI(model_name='anthropic/claude-3.5-haiku', generation_config=api_model_generation_config)
    elif "claude" in model_name:
        return ClaudeAPI(model_name=model_name, generation_config=api_model_generation_config)
    elif model_name == 'gemini1.5-pro':
        return GeminiAPI(model_name='google/gemini-pro-1.5', generation_config=api_model_generation_config)
    elif model_name == 'gemini1.5-flash':
        return GeminiAPI(model_name='google/gemini-flash-1.5', generation_config=api_model_generation_config)
    elif "gemini" in model_name:
        return GeminiAPI(model_name=model_name, generation_config=api_model_generation_config)
    elif model_name == 'qwen2.5-72b':
        return QwenCloudAPI(model_name='qwen/qwen-2.5-72b-instruct', generation_config=api_model_generation_config)
    elif model_name == 'deepseekv2.5':
        return DeepseekAPI(model_name='deepseek/deepseek-chat', generation_config=api_model_generation_config)
    elif model_name == 'llama3.1-70b':
        return LlamaCloudAPI(model_name='meta-llama/llama-3.1-70b-instruct', generation_config=api_model_generation_config)
    elif model_name == 'llama3.1-405b':
        return LlamaCloudAPI(model_name='meta-llama/llama-3.1-405b-instruct', generation_config=api_model_generation_config)
    # local model
    elif model_name == 'qwen2.5-7b':
        model_path = "Qwen/Qwen2.5-7B-Instruct"
        return QwenAPI(model_path=model_path, tokenizer_path=model_path, generation_config=local_model_generation_config)
    elif model_name == 'qwen2.5-14b':
        model_path = "Qwen/Qwen2.5-14B-Instruct"
        return QwenAPI(model_path=model_path, tokenizer_path=model_path, generation_config=local_model_generation_config)
    elif model_name == 'llama3.1-8b':
        model_path = "meta-llama/Llama-3.1-8B-Instruct"
        return Llama3API(model_path=model_path, tokenizer_path=model_path, generation_config=local_model_generation_config)
    elif model_name == 'glm4-9b':
        model_path = "THUDM/glm-4-9b-chat"
        return GLM4API(model_path=model_path, tokenizer_path=model_path, generation_config=local_model_generation_config)
    # vLLM model: format is "vllm:<model_path>@<server_url>"
    # Example: vllm:/data/share/models/qwen2.5/Qwen2.5-32B-Instruct@http://localhost:8000/v1
    elif model_name.startswith('vllm:'):
        vllm_spec = model_name[5:]  # Remove 'vllm:' prefix
        if '@' in vllm_spec:
            vllm_model_path, server_url = vllm_spec.rsplit('@', 1)
        else:
            vllm_model_path = vllm_spec
            server_url = "http://localhost:8000/v1"
        return VllmAPI(model_name=vllm_model_path, generation_config=api_model_generation_config, base_url=server_url)
    else:
        raise ValueError(f"Model {model_name} not found.")

error_log_path = './error_samples.json'
exceed_log_path = './exceed_round_samples.json'

@ray.remote
class EvaluatorWorker:
    """Ray worker class that holds model API and EnvManager instances."""
    def __init__(self, model_name, api_model_generation_config, local_model_generation_config, allow_empty):
        # Set up sys.path for worker process (Ray workers run in separate processes)
        import sys
        import os
        worker_dir = os.path.dirname(os.path.abspath(__file__))
        environments_path = os.path.abspath(os.path.join(worker_dir, '../environments'))
        if environments_path not in sys.path:
            sys.path.insert(0, environments_path)
        
        # Import EnvManager in worker process after path is set
        from EnvManager import EnvManager
        
        self.agent_api = create_agent_api(model_name, api_model_generation_config, local_model_generation_config)
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
                # if rounds > 8:
                #     print(f'current round: {rounds}, messages: {messages}')
                # else:
                #     print(f'id {d["id"]}, current round: {rounds}')
                if rounds > 8:
                    print(f'current round: {rounds}, messages: {messages}')
                else:
                    print(f'id {d["id"]}, current round: {rounds}')

                completion = self.agent_api.generate_response(messages, tool_descs)

                rounds += 1

                if completion is None or 'type' not in completion: continue
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
                        if tool_name not in provided_tools: continue
                        if not env.has_tool(tool_name): continue
                        tool_call_result = env.call_tool(tool_name, deepcopy(arguments))
                        break

                    if tool_call_result is None:
                        tool_call_result = {"success": False, "message": f"Tool {tool_name} doesn't exist."}

                    # Use full_assistant_message if available (for Gemini's thought_signature)
                    # Otherwise construct the message manually (for other APIs)
                    # CRITICAL: Use deepcopy to preserve extra_content with thought_signature
                    if 'full_assistant_message' in completion:
                        tool_call_response = copy.deepcopy(completion['full_assistant_message'])
                    else:
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

                ## normal content part
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
            import traceback; traceback.print_exc();
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
        # print(e)
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

def eval_file(path, outpath, error_log_path, exceed_log_path, model_name, api_model_generation_config, local_model_generation_config, allow_empty, num_workers):
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
    print("data: ", len(outids))
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
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=num_workers, ignore_reinit_error=True)
    
    # Create worker pool
    workers = [EvaluatorWorker.remote(model_name, api_model_generation_config, local_model_generation_config, allow_empty) 
               for _ in range(num_workers)]
    
    # Process data in parallel
    success_count = 0
    fail_count = 0
    exceed_count = 0
    error_samples = []
    exceed_samples = []
    
    # Use round-robin assignment to workers
    worker_idx = 0
    futures = []
    sample_indices = []
    
    for idx, d in enumerate(data):
        worker = workers[worker_idx]
        future = worker.evaluate.remote(d)
        futures.append(future)
        sample_indices.append(idx)
        worker_idx = (worker_idx + 1) % num_workers
    
    # Collect results with progress bar
    results = []
    for future, idx in tqdm(zip(futures, sample_indices), total=len(futures), desc="Evaluating"):
        try:
            result = ray.get(future)
            results.append((idx, result))
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
    for idx, result in results:
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
    
    print(f'success count: {success_count} (exceed count: {exceed_count}), fail count: {fail_count}')
    
    # Shutdown Ray
    ray.shutdown()


if __name__ == '__main__':
    path = "../data/released_data.json"
    outdir = f"./evaluation_results/tot-{model_name}"

    outdir += f'{args.extra_info}'
    outpath = f"{outdir}/gen_res.json"
    error_log_path = f"{outdir}/error_samples.json"
    exceed_log_path = f"{outdir}/exceed_round_samples.json"
    log_path = f"{outdir}/gen_log.txt"
    
    os.makedirs(outdir, exist_ok=True)

    redirect_output(log_path, 'w')
    regenerate_exceed_samples = args.regen_exceed
    if regenerate_exceed_samples:
        remove_exceed_samples(outpath, exceed_log_path)
        clear_log(error_log_path, exceed_log_path, clear_error=True, clear_exceed=True)
    else:
        clear_log(error_log_path, exceed_log_path, clear_error=True, clear_exceed=False)
    
    eval_file(path, outpath, error_log_path, exceed_log_path, model_name, 
              api_model_generation_config, local_model_generation_config, 
              args.allow_empty, num_workers)

