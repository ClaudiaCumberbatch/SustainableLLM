#!/usr/bin/env python3
"""
Multi-GPU Parallel vLLM Profiling Script - Improved Version
Distributes configurations across multiple GPUs for parallel experimentation
"""

import aiohttp
import asyncio
import csv
import itertools
import json
import multiprocessing
import numpy as np
import os
import pandas as pd
import psutil
import queue
import random
import requests
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing import Lock, Manager, Process, Queue
from typing import Any, Dict, List, Optional

# For A100 GPUs
MOCK_MODE = False  # Set to True for testing without GPU

# Mock or real pynvml
if MOCK_MODE:
    class MockNvml:
        @staticmethod
        def nvmlInit():
            print("Mock: NVML initialized")
        
        @staticmethod
        def nvmlShutdown():
            print("Mock: NVML shutdown")
        
        @staticmethod
        def nvmlDeviceGetHandleByIndex(index):
            return f"mock_gpu_{index}"
        
        @staticmethod
        def nvmlDeviceGetPowerUsage(handle):
            return random.randint(200000, 400000)  # A100 power range
        
        @staticmethod
        def nvmlDeviceGetUtilizationRates(handle):
            class Util:
                gpu = random.randint(50, 100)
            return Util()
        
        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle):
            class MemInfo:
                used = random.randint(10, 40) * (1024**3)  # A100 40GB
            return MemInfo()
    
    pynvml = MockNvml
else:
    import pynvml

# Initialize NVIDIA Management Library
pynvml.nvmlInit()

@dataclass
class ProfilingResult:
    """Data class for storing profiling results"""
    # GPU info
    gpu_id: int
    
    # Configuration parameters
    power_cap: int
    gpu_memory_utilization: float
    max_num_seqs: int
    enable_prefix_caching: bool
    enable_chunked_prefill: bool
    swap_space: int
    max_num_batched_tokens: int
    mps_percentage: int
    
    # Request information
    prompt_id: int
    prompt: str
    prompt_length: int
    
    # Performance metrics
    ttft: float
    tbt: float
    total_latency: float
    tokens_generated: int
    throughput: float
    
    # Energy metrics
    energy_kwh: float
    avg_power_kw: float
    peak_power_kw: float
    
    # System metrics
    gpu_utilization: float
    gpu_memory_used_gb: float
    
    # Metadata
    timestamp: str
    generated_text: str

class DatabaseManager:
    """Thread-safe database manager with retry logic"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        # Use WAL mode for better concurrent access
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.create_tables()
        
    def create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profiling_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                -- GPU info
                gpu_id INTEGER,
                
                -- Configuration
                power_cap INTEGER,
                gpu_memory_utilization REAL,
                max_num_seqs INTEGER,
                enable_prefix_caching BOOLEAN,
                enable_chunked_prefill BOOLEAN,
                swap_space INTEGER,
                max_num_batched_tokens INTEGER,
                mps_percentage INTEGER,
                
                -- Request info
                prompt_id INTEGER,
                prompt TEXT,
                prompt_length INTEGER,
                
                -- Performance metrics
                ttft REAL,
                tbt REAL,
                total_latency REAL,
                tokens_generated INTEGER,
                throughput REAL,
                
                -- Energy metrics
                energy_kwh REAL,
                avg_power_kw REAL,
                peak_power_kw REAL,
                
                -- System metrics
                gpu_utilization REAL,
                gpu_memory_used_gb REAL,
                
                -- Metadata
                timestamp TEXT,
                generated_text TEXT,
                
                -- Index
                UNIQUE(gpu_id, prompt_id, power_cap, gpu_memory_utilization, max_num_seqs, 
                       enable_prefix_caching, enable_chunked_prefill, swap_space, 
                       max_num_batched_tokens, mps_percentage)
            )
        ''')
        
        self.conn.commit()
    
    def insert_result(self, result: ProfilingResult, max_retries: int = 3):
        """Thread-safe insert with retry logic"""
        for attempt in range(max_retries):
            try:
                with self.lock:
                    cursor = self.conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO profiling_results (
                            gpu_id, power_cap, gpu_memory_utilization, max_num_seqs,
                            enable_prefix_caching, enable_chunked_prefill,
                            swap_space, max_num_batched_tokens, mps_percentage,
                            prompt_id, prompt, prompt_length,
                            ttft, tbt, total_latency, tokens_generated, throughput,
                            energy_kwh, avg_power_kw, peak_power_kw,
                            gpu_utilization, gpu_memory_used_gb,
                            timestamp, generated_text
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        result.gpu_id, result.power_cap, result.gpu_memory_utilization, result.max_num_seqs,
                        result.enable_prefix_caching, result.enable_chunked_prefill,
                        result.swap_space, result.max_num_batched_tokens, result.mps_percentage,
                        result.prompt_id, result.prompt[:1000], result.prompt_length,  # Limit prompt length
                        result.ttft, result.tbt, result.total_latency, result.tokens_generated, result.throughput,
                        result.energy_kwh, result.avg_power_kw, result.peak_power_kw,
                        result.gpu_utilization, result.gpu_memory_used_gb,
                        result.timestamp, result.generated_text[:1000]  # Limit generated text length
                    ))
                    self.conn.commit()
                    return  # Success
            except sqlite3.Error as e:
                print(f"[GPU {result.gpu_id}] Database error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(0.1, 0.5))  # Random backoff
                else:
                    print(f"[GPU {result.gpu_id}] Failed to insert result after {max_retries} attempts")
                    self.conn.rollback()

class DatasetLoader:
    """Load and manage prompt datasets"""
    
    def __init__(self, prompt_file: str, limit: int = None):
        self.prompt_file = prompt_file
        self.prompts = []
        self.limit = limit
        if "grad_school_math" in prompt_file:
            self.load_math()
        elif "alpaca" in prompt_file:
            self.load_alpaca()
        elif "MMLU" in prompt_file:
            self.load_mmlu()
        elif "NQ" in prompt_file:
            self.load_nq()
        elif "scienceQA" in prompt_file:
            self.load_scienceqa()
    
    def load_math(self):
        """Load prompt data"""
        self.prompts = []
        if os.path.exists(self.prompt_file):
            with open(self.prompt_file, 'r') as f:
                for i, line in enumerate(f):
                    if self.limit and i >= self.limit:
                        break
                    try:
                        data = json.loads(line.strip())
                        if 'question' in data:
                            self.prompts.append(data['question'])
                        elif 'prompt' in data:
                            self.prompts.append(data['prompt'])
                        elif 'text' in data:
                            self.prompts.append(data['text'])
                    except json.JSONDecodeError:
                        continue
            print(f"Loaded {len(self.prompts)} prompts from {self.prompt_file}")
        else:
            print(f"Warning: Prompt file not found: {self.prompt_file}")
            print("Generating mock prompts for testing...")
            self.prompts = [f"What is the capital of country {i}?" for i in range(10)]
            print(f"Generated {len(self.prompts)} mock prompts")

    def load_alpaca(self):
        """Load Alpaca dataset"""
        self.prompts = []
        if os.path.exists(self.prompt_file):
            with open(self.prompt_file, 'r') as f:
                data = json.load(f)
                for i, item in enumerate(data):
                    if self.limit and i >= self.limit:
                        break
                    if 'instruction' in item:
                        instruction = item['instruction']
                        input_text = item['input']
                        prompt = f"""Instruction: {instruction}
                        Input: {input_text}
                        Response:"""
                        self.prompts.append(prompt)
            print(f"Loaded {len(self.prompts)} prompts from {self.prompt_file}")
        else:
            print(f"Warning: Prompt file not found: {self.prompt_file}")
            print("Generating mock prompts for testing...")
            self.prompts = [f"Explain the concept of gravity in simple terms. (Prompt {i})" for i in range(10)]
            print(f"Generated {len(self.prompts)} mock prompts")
    
    # csv version
    # def load_mmlu(self):
    #     """Load MMLU dataset"""
    #     if os.path.exists(self.prompt_file):
    #         with open(self.prompt_file, 'r', encoding='utf-8') as f:
    #             reader = csv.DictReader(f)
    #             for row in reader:
    #                 self.prompts.append(row['prompt'])
    #                 if self.limit and len(self.prompts) >= self.limit:
    #                     break
    #         print(f"Loaded {len(self.prompts)} prompts from {self.prompt_file}")
    #     else:
    #         print(f"Warning: Prompt file not found: {self.prompt_file}")
    #         print("Generating mock prompts for testing...")
    #         self.prompts = [f"What is the derivative of x^2? (Prompt {i})" for i in range(10)]
    #         print(f"Generated {len(self.prompts)} mock prompts")

    # json version
    def load_mmlu(self):
        """Load MMLU dataset"""
        if os.path.exists(self.prompt_file):
            with open(self.prompt_file, 'r') as f:
                data = json.load(f)
                for i, item in enumerate(data):
                    if self.limit and i >= self.limit:
                        break
                    if 'prompt' in item:
                        self.prompts.append(item['prompt'])
            print(f"Loaded {len(self.prompts)} prompts from {self.prompt_file}")
        else:
            print(f"Warning: Prompt file not found: {self.prompt_file}")
            print("Generating mock prompts for testing...")
            self.prompts = [f"What is the capital of country {i}?" for i in range(10)]
            print(f"Generated {len(self.prompts)} mock prompts")
    
    def load_nq(self):
        """Load Natural Questions dataset"""
        self.prompts = []
        if os.path.exists(self.prompt_file):
            with open(self.prompt_file, 'r') as f:
                for i, line in enumerate(f):
                    if self.limit and i >= self.limit:
                        break

                    self.prompts.append(line.strip())

            print(f"Loaded {len(self.prompts)} prompts from {self.prompt_file}")
        else:
            print(f"Warning: Prompt file not found: {self.prompt_file}")
            print("Generating mock prompts for testing...")
            self.prompts = [f"What is the capital of country {i}?" for i in range(10)]
            print(f"Generated {len(self.prompts)} mock prompts")

    def load_scienceqa(self):
        """Load ScienceQA dataset"""
        self.prompts = []
        if os.path.exists(self.prompt_file):
            with open(self.prompt_file, 'r') as f:
                data = json.load(f)
                for i, (key, item) in enumerate(data.items()):
                    if self.limit and len(self.prompts) >= self.limit:
                        break
                    if 'prompt' in item:
                        self.prompts.append(item['prompt'])
            print(f"Loaded {len(self.prompts)} prompts from {self.prompt_file}")
        else:
            print(f"Warning: Prompt file not found: {self.prompt_file}")
            print("Generating mock prompts for testing...")
            self.prompts = [f"What is the boiling point of water? (Prompt {i})" for i in range(10)]
            print(f"Generated {len(self.prompts)} mock prompts")

    def get_all_prompts(self) -> List[str]:
        return self.prompts

def calculate_energy_metrics(power_readings: List[Dict]) -> Dict[str, float]:
    """Calculate energy consumption from power readings"""
    if not power_readings:
        return {
            'energy_kwh': 0, 
            'avg_power_kw': 0, 
            'peak_power_kw': 0,
            'avg_gpu_utilization': 0,
            'avg_memory_used_gb': 0
        }
    
    powers = [r['power_w'] for r in power_readings]
    avg_power_w = np.mean(powers)
    peak_power_w = np.max(powers)
    
    # Calculate energy
    energy_j = 0
    for i in range(1, len(power_readings)):
        dt = power_readings[i]['timestamp'] - power_readings[i-1]['timestamp']
        avg_p = (power_readings[i]['power_w'] + power_readings[i-1]['power_w']) / 2
        energy_j += avg_p * dt
    
    return {
        'energy_kwh': energy_j / (3600 * 1000),
        'avg_power_kw': avg_power_w / 1000.0,
        'peak_power_kw': peak_power_w / 1000.0,
        'avg_gpu_utilization': np.mean([r['gpu_utilization'] for r in power_readings]),
        'avg_memory_used_gb': np.mean([r['memory_used_gb'] for r in power_readings])
    }

def cleanup_gpu_processes(gpu_id: int):
    """Clean up any hanging processes for a GPU"""
    try:
        # Kill any vllm processes on this GPU
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and 'vllm' in str(cmdline) and f'CUDA_VISIBLE_DEVICES={gpu_id}' in str(cmdline):
                    proc.terminate()
                    proc.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass
    except Exception as e:
        print(f"[GPU {gpu_id}] Error during cleanup: {e}")

def run_single_gpu_experiment(gpu_id: int, configs: List[Dict], prompts: List[str], 
                             port_base: int, db_path: str, output_dir: str,
                             model_name: str, result_queue: Queue):
    """Run experiments on a single GPU with improved error handling"""
    
    # Setup signal handler for clean shutdown
    def signal_handler(signum, frame):
        print(f"[GPU {gpu_id}] Received signal {signum}, cleaning up...")
        cleanup_gpu_processes(gpu_id)
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"\n[GPU {gpu_id}] Starting experiments with {len(configs)} configurations")
    
    # Initialize components for this GPU
    from simplified_profiler import VLLMProfiler, PowerMonitor
    
    port = port_base + gpu_id
    db_manager = DatabaseManager(db_path)
    
    server_process = None
    
    for config_idx, config in enumerate(configs):
        print(f"\n[GPU {gpu_id}] Running configuration {config_idx + 1}/{len(configs)}")
        print(f"[GPU {gpu_id}] Config: {config}")
        
        try:
            # Clean up before starting
            cleanup_gpu_processes(gpu_id)
            time.sleep(2)
            
            # Set power cap for this specific GPU
            if not MOCK_MODE:
                power_monitor = PowerMonitor(gpu_index=gpu_id)
                
                cmd = f"sudo nvidia-smi -i {gpu_id} -pl {config['power_cap']}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"[GPU {gpu_id}] Warning: Failed to set power cap: {result.stderr}")
                
                # Set CUDA MPS active thread percentage
                percentage = config['mps_percentage']
                os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(percentage)
                print(f"[GPU {gpu_id}] MPS active thread percentage set to {percentage}%")
            
            # Start vLLM server on this GPU
            server_process = start_vllm_server_for_gpu(
                gpu_id, port, model_name, config, output_dir
            )
            
            if not server_process:
                print(f"[GPU {gpu_id}] Failed to start server, skipping configuration")
                continue
            
            # Process all prompts
            consecutive_failures = 0
            max_consecutive_failures = 5
            
            for prompt_idx, prompt in enumerate(prompts):
                if consecutive_failures >= max_consecutive_failures:
                    print(f"[GPU {gpu_id}] Too many consecutive failures, skipping remaining prompts")
                    break
                
                print(f"[GPU {gpu_id}] Processing prompt {prompt_idx + 1}/{len(prompts)}")
                
                try:
                    if not MOCK_MODE:
                        power_monitor.start()
                    
                    # Run inference with timeout
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    result = loop.run_until_complete(
                        run_inference_request(gpu_id, port, prompt, prompt_idx, model_name)
                    )
                    loop.close()
                    
                    if not MOCK_MODE:
                        power_readings = power_monitor.stop()
                        energy_metrics = calculate_energy_metrics(power_readings)
                    else:
                        energy_metrics = {
                            'energy_kwh': 0.001,
                            'avg_power_kw': 0.3,
                            'peak_power_kw': 0.4,
                            'avg_gpu_utilization': 80,
                            'avg_memory_used_gb': 25
                        }
                    
                    # Create ProfilingResult
                    profiling_result = ProfilingResult(
                        gpu_id=gpu_id,
                        power_cap=config['power_cap'],
                        gpu_memory_utilization=config['gpu_memory_utilization'],
                        max_num_seqs=config['max_num_seqs'],
                        enable_prefix_caching=config['enable_prefix_caching'],
                        enable_chunked_prefill=config['enable_chunked_prefill'],
                        swap_space=config['swap_space'],
                        max_num_batched_tokens=config['max_num_batched_tokens'],
                        mps_percentage=config['mps_percentage'],
                        prompt_id=prompt_idx,
                        prompt=prompt,
                        prompt_length=len(prompt.split()),
                        ttft=result['ttft'],
                        tbt=result['tbt'],
                        total_latency=result['total_latency'],
                        tokens_generated=result['tokens_generated'],
                        throughput=result['throughput'],
                        energy_kwh=energy_metrics['energy_kwh'],
                        avg_power_kw=energy_metrics['avg_power_kw'],
                        peak_power_kw=energy_metrics['peak_power_kw'],
                        gpu_utilization=energy_metrics['avg_gpu_utilization'],
                        gpu_memory_used_gb=energy_metrics['avg_memory_used_gb'],
                        timestamp=datetime.now().isoformat(),
                        generated_text=result['generated_text']
                    )
                    
                    # Save to database
                    db_manager.insert_result(profiling_result)
                    
                    # Put result in queue for progress tracking
                    result_queue.put({
                        'gpu_id': gpu_id,
                        'config_idx': config_idx,
                        'prompt_idx': prompt_idx,
                        'total_configs': len(configs),
                        'total_prompts': len(prompts)
                    })
                    
                    consecutive_failures = 0  # Reset failure counter on success
                    
                except Exception as e:
                    print(f"[GPU {gpu_id}] Error processing prompt {prompt_idx}: {e}")
                    consecutive_failures += 1
                    # Log error but continue
                    continue
            
            # Stop server
            if server_process:
                try:
                    server_process.terminate()
                    server_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server_process.kill()
                    server_process.wait()
                server_process = None
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error with configuration {config_idx}: {e}")
            traceback.print_exc()
            
            # Clean up server if it exists
            if server_process:
                try:
                    server_process.kill()
                except:
                    pass
                server_process = None
            
            cleanup_gpu_processes(gpu_id)
            continue
        
        # Cool down between configurations
        cleanup_gpu_processes(gpu_id)
        time.sleep(5)
    
    print(f"[GPU {gpu_id}] Completed all experiments")

def start_vllm_server_for_gpu(gpu_id: int, port: int, model_name: str, 
                              config: Dict, output_dir: str):
    """Start vLLM server for specific GPU with better error handling"""
    if MOCK_MODE:
        print(f"[GPU {gpu_id}] Mock: Starting vLLM server on port {port}")
        return None
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--gpu-memory-utilization", str(config['gpu_memory_utilization']),
        "--max-num-seqs", str(config['max_num_seqs']),
        "--swap-space", str(config['swap_space']),
        "--max-num-batched-tokens", str(config['max_num_batched_tokens']),
        "--max-model-len", "512",
        "--enforce-eager"
    ]
    
    if config['enable_prefix_caching']:
        cmd.append("--enable-prefix-caching")
    
    if config['enable_chunked_prefill']:
        cmd.append("--enable-chunked-prefill")
    
    log_file = os.path.join(output_dir, "logs", f"gpu_{gpu_id}_server_{time.time()}.log")
    
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd, 
                stdout=f, 
                stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_id)},
                preexec_fn=os.setsid  # Create new process group for clean termination
            )
        
        # Wait for server to be ready with timeout
        if not check_server_health(port, max_retries=60, retry_delay=2):
            print(f"[GPU {gpu_id}] Server failed to start, killing process")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except:
                process.kill()
            raise RuntimeError(f"Server on GPU {gpu_id} failed to start")
        
        print(f"[GPU {gpu_id}] Server started successfully on port {port}")
        return process
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error starting server: {e}")
        return None

def check_server_health(port: int, max_retries: int = 60, retry_delay: int = 2):
    """Check if server is ready"""
    url = f"http://localhost:{port}/health"
    
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(retry_delay)
    
    return False

async def run_inference_request(gpu_id: int, port: int, prompt: str, 
                               prompt_idx: int, model_name: str) -> Dict:
    """Run single inference request with improved error handling"""
    if MOCK_MODE:
        # Return mock results
        return {
            'ttft': random.uniform(0.1, 0.5),
            'tbt': random.uniform(0.01, 0.05),
            'total_latency': random.uniform(1.0, 3.0),
            'tokens_generated': random.randint(50, 150),
            'throughput': random.uniform(30, 100),
            'generated_text': f"Mock response for prompt {prompt_idx}"
        }
    
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": True
    }
    
    result = {
        'ttft': 0,
        'tbt': 0,
        'total_latency': 0,
        'throughput': 0,
        'tokens_generated': 0,
        'generated_text': "",
        'inter_token_times': []
    }
    
    start_time = time.time()
    first_token_time = None
    prev_token_time = start_time
    
    # Create session with custom timeout
    timeout = aiohttp.ClientTimeout(total=300, connect=30, sock_read=60)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Server returned status {response.status}: {error_text}")
                
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            current_time = time.time()
                            data_str = line_str[6:]
                            
                            if data_str != '[DONE]':
                                try:
                                    data = json.loads(data_str)
                                    if 'choices' in data and len(data['choices']) > 0:
                                        choice = data['choices'][0]
                                        if 'text' in choice:
                                            result['generated_text'] += choice['text']
                                            result['tokens_generated'] += 1
                                            
                                            if first_token_time is None:
                                                first_token_time = current_time
                                                result['ttft'] = first_token_time - start_time
                                            else:
                                                result['inter_token_times'].append(
                                                    current_time - prev_token_time
                                                )
                                            
                                            prev_token_time = current_time
                                except json.JSONDecodeError as e:
                                    print(f"[GPU {gpu_id}] JSON decode error: {e}, data: {data_str[:100]}")
                                    continue
            
            result['total_latency'] = time.time() - start_time
            result['throughput'] = (result['tokens_generated'] / result['total_latency']
                                  if result['total_latency'] > 0 else 0)
            
            if result['inter_token_times']:
                result['tbt'] = np.mean(result['inter_token_times'])
            
        except asyncio.TimeoutError:
            print(f"[GPU {gpu_id}] Timeout during inference for prompt {prompt_idx}")
            result['total_latency'] = time.time() - start_time
            # Return partial results if available
        except Exception as e:
            print(f"[GPU {gpu_id}] Error during inference for prompt {prompt_idx}: {e}")
            result['total_latency'] = time.time() - start_time
            # Return partial results if available
    
    return result

class MultiGPUProfiler:
    """Main class for multi-GPU parallel profiling with improved robustness"""
    
    def __init__(self, num_gpus: int = 4, model_name="meta-llama/Llama-2-7b-hf",
                 prompt_file="../dataset/grad_school_math/train.jsonl", output_dir=None, prompt_limit=None):
        self.num_gpus = num_gpus
        self.model_name = model_name
        self.port_base = 8000  # Each GPU gets port 8000, 8001, 8002, 8003
        
        # Load prompts
        self.dataset_loader = DatasetLoader(prompt_file, limit=prompt_limit)
        self.prompts = self.dataset_loader.get_all_prompts()
        
        # Create output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"multi_gpu_profiling_{timestamp}"
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        
        # Database path (shared across GPUs)
        self.db_path = os.path.join(self.output_dir, "profiling_results.db")
        

        # Test
        # self.param_ranges = {
        #     'power_cap': [250, 300],  # V100 power range
        #     'gpu_memory_utilization': [0.80, 0.90],
        #     'max_num_seqs': [256],
        #     'enable_prefix_caching': [True],
        #     'enable_chunked_prefill': [False],
        #     'swap_space': [4],
        #     'max_num_batched_tokens': [4096],
        #     'mps_percentage': [75, 100]
        # }  

        # Parameter ranges optimized for A100 
        # self.param_ranges = {
        #     'power_cap': [300, 350, 400],  # A100 power range
        #     'gpu_memory_utilization': [0.85, 0.90, 0.95],
        #     'max_num_seqs': [64, 128, 256],
        #     'enable_prefix_caching': [True, False],
        #     'enable_chunked_prefill': [True, False],
        #     'swap_space': [4, 8],
        #     'max_num_batched_tokens': [4096, 8192],
        #     'mps_percentage': [50, 75, 100]
        # }

        # Parameter ranges optimized for V100 
        self.param_ranges = {
            'power_cap': [200, 250, 300],  # V100 power range
            'gpu_memory_utilization': [0.85, 0.90, 0.95],
            'max_num_seqs': [64, 128, 256],
            'enable_prefix_caching': [True, False],
            'enable_chunked_prefill': [False],
            'swap_space': [4, 8],
            'max_num_batched_tokens': [4096, 8192],
            'mps_percentage': [50, 75, 100]
        }
        
        # Save configuration
        config_file = os.path.join(self.output_dir, "experiment_config.json")
        with open(config_file, 'w') as f:
            json.dump({
                'num_gpus': num_gpus,
                'model_name': model_name,
                'prompt_file': prompt_file,
                'param_ranges': self.param_ranges,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
    
    def generate_all_configs(self) -> List[Dict]:
        """Generate all parameter combinations"""
        all_configs = []
        for config_tuple in itertools.product(*[
            self.param_ranges['power_cap'],
            self.param_ranges['gpu_memory_utilization'],
            self.param_ranges['max_num_seqs'],
            self.param_ranges['enable_prefix_caching'],
            self.param_ranges['enable_chunked_prefill'],
            self.param_ranges['swap_space'],
            self.param_ranges['max_num_batched_tokens'],
            self.param_ranges['mps_percentage']
        ]):
            all_configs.append({
                'power_cap': config_tuple[0],
                'gpu_memory_utilization': config_tuple[1],
                'max_num_seqs': config_tuple[2],
                'enable_prefix_caching': config_tuple[3],
                'enable_chunked_prefill': config_tuple[4],
                'swap_space': config_tuple[5],
                'max_num_batched_tokens': config_tuple[6],
                'mps_percentage': config_tuple[7]
            })
        return all_configs
    
    def distribute_configs(self, all_configs: List[Dict]) -> Dict[int, List[Dict]]:
        """Distribute configurations evenly across GPUs"""
        gpu_configs = {i: [] for i in range(self.num_gpus)}
        
        for idx, config in enumerate(all_configs):
            gpu_id = idx % self.num_gpus
            gpu_configs[gpu_id].append(config)
        
        return gpu_configs
    
    def run_parallel_experiments(self):
        """Run experiments in parallel across all GPUs with improved monitoring"""
        print(f"\n{'='*70}")
        print(f"STARTING MULTI-GPU PARALLEL EXPERIMENTS")
        print(f"Number of GPUs: {self.num_gpus}")
        print(f"Number of prompts: {len(self.prompts)}")
        print(f"{'='*70}")
        
        # Clean up any existing processes before starting
        for gpu_id in range(self.num_gpus):
            cleanup_gpu_processes(gpu_id)
        
        # Generate and distribute configurations
        all_configs = self.generate_all_configs()
        print(f"Total configurations: {len(all_configs)}")
        
        gpu_configs = self.distribute_configs(all_configs)
        for gpu_id, configs in gpu_configs.items():
            print(f"GPU {gpu_id}: {len(configs)} configurations")
        
        # Create shared result queue for progress tracking
        manager = Manager()
        result_queue = manager.Queue()
        
        # Start processes for each GPU
        processes = []
        for gpu_id in range(self.num_gpus):
            p = Process(
                target=run_single_gpu_experiment,
                args=(
                    gpu_id,
                    gpu_configs[gpu_id],
                    self.prompts,
                    self.port_base,
                    self.db_path,
                    self.output_dir,
                    self.model_name,
                    result_queue
                )
            )
            p.start()
            processes.append(p)
            print(f"Started process for GPU {gpu_id} (PID: {p.pid})")
        
        # Monitor progress with better timeout handling
        total_experiments = sum(len(configs) * len(self.prompts) 
                              for configs in gpu_configs.values())
        completed = 0
        
        print(f"\nTotal experiments to run: {total_experiments}")
        print("Progress:")
        
        last_completed = completed
        last_update_time = time.time()
        timeout_sec = 300
        stall_count = 0
        max_stall_count = 3
        
        while completed < total_experiments:
            try:
                # Non-blocking get with timeout
                result = result_queue.get(timeout=30)
                completed += 1
                
                if completed != last_completed:
                    last_update_time = time.time()
                    last_completed = completed
                    stall_count = 0  # Reset stall counter
                
                # Update progress
                if completed % 10 == 0 or completed == total_experiments:
                    progress = (completed / total_experiments) * 100
                    elapsed = time.time() - last_update_time
                    print(f"  [{progress:.1f}%] Completed {completed}/{total_experiments} experiments (last update: {elapsed:.1f}s ago)")
                    
            except queue.Empty:
                # Check process status
                alive_processes = sum(1 for p in processes if p.is_alive())
                
                if alive_processes == 0:
                    print("\nAll processes have terminated")
                    break
                
                # Check for stall
                elapsed = time.time() - last_update_time
                if elapsed > timeout_sec:
                    stall_count += 1
                    print(f"\nWarning: No progress for {elapsed:.1f} seconds (stall count: {stall_count}/{max_stall_count})")
                    
                    if stall_count >= max_stall_count:
                        print(f"Maximum stall count reached. Terminating remaining processes...")
                        for p in processes:
                            if p.is_alive():
                                print(f"  Terminating process {p.pid}")
                                p.terminate()
                                p.join(timeout=10)
                                if p.is_alive():
                                    p.kill()
                        break
                    
                    last_update_time = time.time()  # Reset timer after warning
                
                # Log process status
                print(f"  Active processes: {alive_processes}/{self.num_gpus}")
                for i, p in enumerate(processes):
                    if p.is_alive():
                        print(f"    GPU {i}: Running (PID {p.pid})")
                    else:
                        print(f"    GPU {i}: Terminated (exit code: {p.exitcode})")
            
            except KeyboardInterrupt:
                print("\nInterrupted by user. Cleaning up...")
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                break
            
            except Exception as e:
                print(f"Unexpected error in monitoring loop: {e}")
                continue
        
        # Wait for all processes to complete or timeout
        print("\nWaiting for processes to complete...")
        for i, p in enumerate(processes):
            if p.is_alive():
                print(f"  Waiting for GPU {i} process (PID {p.pid})...")
                p.join(timeout=30)
                if p.is_alive():
                    print(f"  Force terminating GPU {i} process...")
                    p.terminate()
                    p.join(timeout=10)
                    if p.is_alive():
                        p.kill()
        
        # Final cleanup
        for gpu_id in range(self.num_gpus):
            cleanup_gpu_processes(gpu_id)
        
        print(f"\n{'='*70}")
        print("EXPERIMENT RUN COMPLETED")
        print(f"Completed {completed}/{total_experiments} experiments")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*70}")
        
        # Export results to CSV
        self.export_results_to_csv()
    
    def export_results_to_csv(self):
        """Export database results to CSV files with error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if there are any results
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM profiling_results")
            count = cursor.fetchone()[0]
            
            if count == 0:
                print("No results found in database")
                conn.close()
                return
            
            print(f"Found {count} results in database")
            
            # Export detailed results
            df_results = pd.read_sql_query("SELECT * FROM profiling_results", conn)
            csv_path = os.path.join(self.output_dir, "detailed_results.csv")
            df_results.to_csv(csv_path, index=False)
            print(f"Detailed results exported to: {csv_path}")
            
            # Generate summary statistics per GPU
            if not df_results.empty:
                summary_path = os.path.join(self.output_dir, "gpu_summary.csv")
                df_summary = df_results.groupby('gpu_id').agg({
                    'ttft': ['mean', 'std', 'min', 'max'],
                    'throughput': ['mean', 'std', 'min', 'max'],
                    'energy_kwh': ['mean', 'sum'],
                    'gpu_utilization': ['mean']
                }).round(4)
                df_summary.to_csv(summary_path)
                print(f"GPU summary exported to: {summary_path}")
                
                # Generate configuration summary
                config_summary_path = os.path.join(self.output_dir, "config_summary.csv")
                df_config_summary = df_results.groupby([
                    'power_cap', 'gpu_memory_utilization', 'max_num_seqs',
                    'enable_prefix_caching', 'enable_chunked_prefill',
                    'swap_space', 'max_num_batched_tokens', 'mps_percentage'
                ]).agg({
                    'throughput': ['mean', 'std'],
                    'energy_kwh': ['mean', 'std'],
                    'ttft': ['mean', 'std'],
                    'gpu_utilization': ['mean']
                }).round(4)
                df_config_summary.to_csv(config_summary_path)
                print(f"Configuration summary exported to: {config_summary_path}")
            
            conn.close()
            
        except Exception as e:
            print(f"Error exporting results: {e}")
            traceback.print_exc()


def main():
    """Main execution function with improved cleanup"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-GPU Parallel vLLM Profiling')
    parser.add_argument('--num-gpus', type=int, default=4,
                       help='Number of GPUs to use (default: 4)')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                       help='Model name to profile')
    parser.add_argument('--prompt-file', type=str, default='../dataset/grad_school_math/train.jsonl',
                       help='Path to prompt JSONL file')
    parser.add_argument('--prompt-limit', type=int, default=None,
                       help='Limit number of prompts to use')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--mock', action='store_true',
                       help='Run in mock mode without GPU')
    
    args = parser.parse_args()
    
    # Set mock mode
    global MOCK_MODE
    MOCK_MODE = args.mock
    
    if MOCK_MODE:
        print("\n" + "="*70)
        print(" RUNNING IN MOCK MODE (NO GPU REQUIRED) ".center(70))
        print("="*70)
    else:
        print("\n" + "="*70)
        print(f" RUNNING ON {args.num_gpus} GPUs ".center(70))
        print("="*70)
    
    # Clean up any existing vLLM processes before starting
    print("Cleaning up any existing processes...")
    subprocess.run("pkill -f vllm.entrypoints.openai.api_server", shell=True)
    time.sleep(2)
    
    # Initialize multi-GPU profiler
    profiler = MultiGPUProfiler(
        num_gpus=args.num_gpus,
        model_name=args.model,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir,
        prompt_limit=args.prompt_limit
    )
    
    try:
        # Run parallel experiments
        profiler.run_parallel_experiments()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        # Clean up all processes
        subprocess.run("pkill -f vllm.entrypoints.openai.api_server", shell=True)
        for gpu_id in range(args.num_gpus):
            cleanup_gpu_processes(gpu_id)
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        # Clean up all processes
        subprocess.run("pkill -f vllm.entrypoints.openai.api_server", shell=True)
        for gpu_id in range(args.num_gpus):
            cleanup_gpu_processes(gpu_id)
            
    finally:
        if not MOCK_MODE:
            pynvml.nvmlShutdown()
        
        print("\nFinal cleanup...")
        subprocess.run("pkill -f vllm.entrypoints.openai.api_server", shell=True)
        print("Done.")

if __name__ == "__main__":
    main()