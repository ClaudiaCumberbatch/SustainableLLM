#!/usr/bin/env python3
"""
Multi-GPU Parallel vLLM Profiling Script
Distributes configurations across multiple GPUs for parallel experimentation
"""

import os
import sys
import time
import json
import subprocess
import itertools
import pandas as pd
from datetime import datetime
import numpy as np
import asyncio
import aiohttp
from dataclasses import dataclass, asdict
import threading
import queue
import requests
import sqlite3
import random
from typing import Dict, List, Any, Optional
from multiprocessing import Process, Queue, Manager
import multiprocessing
import traceback

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
    """Thread-safe database manager for profiling results"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
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
    
    def insert_result(self, result: ProfilingResult):
        """Thread-safe insert"""
        with self.lock:
            cursor = self.conn.cursor()
            try:
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
                    result.prompt_id, result.prompt, result.prompt_length,
                    result.ttft, result.tbt, result.total_latency, result.tokens_generated, result.throughput,
                    result.energy_kwh, result.avg_power_kw, result.peak_power_kw,
                    result.gpu_utilization, result.gpu_memory_used_gb,
                    result.timestamp, result.generated_text
                ))
                self.conn.commit()
            except sqlite3.Error as e:
                print(f"Database error: {e}")
                self.conn.rollback()

class DatasetLoader:
    """Load and manage prompt datasets"""
    
    def __init__(self, prompt_file: str, limit: int = None):
        self.prompt_file = prompt_file
        self.prompts = []
        self.limit = limit
        self.load_data()
    
    def load_data(self):
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

def run_single_gpu_experiment(gpu_id: int, configs: List[Dict], prompts: List[str], 
                             port_base: int, db_path: str, output_dir: str,
                             model_name: str, result_queue: Queue):
    """Run experiments on a single GPU"""
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"\n[GPU {gpu_id}] Starting experiments with {len(configs)} configurations")
    
    # Initialize components for this GPU
    from simplified_profiler import VLLMProfiler, PowerMonitor
    
    port = port_base + gpu_id
    db_manager = DatabaseManager(db_path)
    
    for config_idx, config in enumerate(configs):
        print(f"\n[GPU {gpu_id}] Running configuration {config_idx + 1}/{len(configs)}")
        print(f"[GPU {gpu_id}] Config: {config}")
        
        try:
            # Set power cap for this specific GPU
            if not MOCK_MODE:
                power_monitor = PowerMonitor(gpu_index=gpu_id)

                cmd = f"sudo nvidia-smi -i {gpu_id} -pl {config['power_cap']}"
                subprocess.run(cmd, shell=True, capture_output=True)

                """Set CUDA MPS active thread percentage"""
                percentage = config['mps_percentage']
                os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(percentage)
                print(f"[GPU {gpu_id}] MPS active thread percentage set to {percentage}%")
            
            # Start vLLM server on this GPU
            server_process = start_vllm_server_for_gpu(
                gpu_id, port, model_name, config, output_dir
            )
            
            # Process all prompts
            for prompt_idx, prompt in enumerate(prompts):
                print(f"[GPU {gpu_id}] Processing prompt {prompt_idx + 1}/{len(prompts)}")
                
                power_monitor.start()

                # Run inference
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(
                    run_inference_request(gpu_id, port, prompt, prompt_idx, model_name)
                )
                loop.close()

                power_readings = power_monitor.stop()
                energy_metrics = calculate_energy_metrics(power_readings)
                
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
            
            # Stop server
            if server_process:
                server_process.terminate()
                server_process.wait()
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error with configuration {config_idx}: {e}")
            traceback.print_exc()  # 打印完整的错误堆栈信息
            subprocess.run("pkill -f vllm.entrypoints.openai.api_server", shell=True)
            continue
        
        # Cool down between configurations
        subprocess.run("pkill -f vllm.entrypoints.openai.api_server", shell=True)
        time.sleep(3)
    
    print(f"[GPU {gpu_id}] Completed all experiments")

def start_vllm_server_for_gpu(gpu_id: int, port: int, model_name: str, 
                              config: Dict, output_dir: str):
    """Start vLLM server for specific GPU"""
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
    with open(log_file, 'w') as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, 
                                  env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_id)})
    
    # Wait for server to be ready
    if not check_server_health(port):
        raise RuntimeError(f"Server on GPU {gpu_id} failed to start")
    
    return process

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
    """Run single inference request"""
    if MOCK_MODE:
        # Return mock results
        return {
            'ttft': random.uniform(0.1, 0.5),
            'tbt': random.uniform(0.01, 0.05),
            'total_latency': random.uniform(1.0, 3.0),
            'tokens_generated': random.randint(50, 150),
            'throughput': random.uniform(30, 100),
            'energy_kwh': random.uniform(0.001, 0.005),
            'avg_power_kw': random.uniform(0.2, 0.4),
            'peak_power_kw': random.uniform(0.3, 0.5),
            'gpu_utilization': random.uniform(50, 100),
            'gpu_memory_used_gb': random.uniform(20, 35),
            'generated_text': f"Mock response for prompt {prompt_idx}"
        }
    
    # Real inference request implementation
    # (simplified version - you'd implement the full streaming logic here)
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

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload) as response:
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
                                except json.JSONDecodeError:
                                    pass
                
            result['total_latency'] = time.time() - start_time
            result['throughput'] = (result['tokens_generated'] / result['total_latency']
                                  if result['total_latency'] > 0 else 0)
                
            if result['inter_token_times']:
                result['tbt'] = np.mean(result['inter_token_times'])
                
        except Exception as e:
            print(f"Error during inference for prompt {prompt_idx}: {e}")
        
    return result


class MultiGPUProfiler:
    """Main class for multi-GPU parallel profiling"""
    
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
        """Run experiments in parallel across all GPUs"""
        print(f"\n{'='*70}")
        print(f"STARTING MULTI-GPU PARALLEL EXPERIMENTS")
        print(f"Number of GPUs: {self.num_gpus}")
        print(f"Number of prompts: {len(self.prompts)}")
        print(f"{'='*70}")
        
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
            print(f"Started process for GPU {gpu_id}")
        
        # Monitor progress
        total_experiments = sum(len(configs) * len(self.prompts) 
                              for configs in gpu_configs.values())
        completed = 0
        
        print(f"\nTotal experiments to run: {total_experiments}")
        print("Progress:")
        
        last_completed = completed
        last_update_time = time.time()
        timeout_sec = 300
        while completed < total_experiments:
            try:
                result = result_queue.get(timeout=1)
                completed += 1
                if completed != last_completed:
                    last_update_time = time.time()
                    last_completed = completed
                if completed % 10 == 0 or completed == total_experiments:
                    progress = (completed / total_experiments) * 100
                    print(f"  [{progress:.1f}%] Completed {completed}/{total_experiments} experiments")
            except:
                # Check if processes are still alive
                if not any(p.is_alive() for p in processes):
                    break
                # 超时检测
                if time.time() - last_update_time > timeout_sec:
                    print(f"No new completed results in {timeout_sec} seconds. Terminating all processes...")
                    for p in processes:
                        if p.is_alive():
                            p.terminate()
                    break
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        print(f"\n{'='*70}")
        print("ALL EXPERIMENTS COMPLETED")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*70}")
        
        # Export results to CSV
        self.export_results_to_csv()
    
    def export_results_to_csv(self):
        """Export database results to CSV files"""
        conn = sqlite3.connect(self.db_path)
        
        # Export detailed results
        df_results = pd.read_sql_query("SELECT * FROM profiling_results", conn)
        csv_path = os.path.join(self.output_dir, "detailed_results.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"Detailed results exported to: {csv_path}")
        
        # Generate summary statistics per GPU
        summary_path = os.path.join(self.output_dir, "gpu_summary.csv")
        df_summary = df_results.groupby('gpu_id').agg({
            'ttft': ['mean', 'std', 'min', 'max'],
            'throughput': ['mean', 'std', 'min', 'max'],
            'energy_kwh': ['mean', 'sum'],
            'gpu_utilization': ['mean']
        }).round(4)
        df_summary.to_csv(summary_path)
        print(f"GPU summary exported to: {summary_path}")
        
        conn.close()


def main():
    """Main execution function"""
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
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if not MOCK_MODE:
            pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()