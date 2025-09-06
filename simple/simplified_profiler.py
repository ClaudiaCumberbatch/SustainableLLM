#!/usr/bin/env python3
"""
Simplified vLLM Inference Profiling Script
- No trace timing dependency, just prompts
- Mock mode for testing without GPU resources
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

# Flag for mock mode (no GPU required)
MOCK_MODE = False  # Set to False when running with actual GPU

# Mock pynvml for testing without GPU
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
            return random.randint(100000, 250000)  # Random power in milliwatts
        
        @staticmethod
        def nvmlDeviceGetUtilizationRates(handle):
            class Util:
                gpu = random.randint(50, 100)
            return Util()
        
        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle):
            class MemInfo:
                used = random.randint(4, 12) * (1024**3)  # Random 4-12 GB
            return MemInfo()
    
    pynvml = MockNvml
else:
    import pynvml

# Initialize NVIDIA Management Library
pynvml.nvmlInit()

@dataclass
class ProfilingResult:
    """Data class for storing profiling results"""
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
    prompt_id: int  # Simplified: just track prompt index
    prompt: str
    prompt_length: int
    
    # Performance metrics
    ttft: float  # Time to first token
    tbt: float   # Time between tokens
    total_latency: float  # Total request latency
    tokens_generated: int
    throughput: float  # Tokens per second
    
    # Energy metrics
    energy_kwh: float  # Energy consumption in kWh
    avg_power_kw: float  # Average power in kW
    peak_power_kw: float  # Peak power in kW
    
    # System metrics
    gpu_utilization: float
    gpu_memory_used_gb: float
    
    # Metadata
    timestamp: str
    generated_text: str

class DatabaseManager:
    """Manage SQLite database for profiling results"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
        
    def create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        # Simplified profiling results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profiling_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                -- Configuration
                power_cap INTEGER,
                gpu_memory_utilization REAL,
                max_num_seqs INTEGER,
                enable_prefix_caching BOOLEAN,
                enable_chunked_prefill BOOLEAN,
                swap_space INTEGER,
                max_num_batched_tokens INTEGER,
                mps_percentage INTEGER,
                
                -- Request info (simplified)
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
                
                -- Index for queries
                UNIQUE(prompt_id, power_cap, gpu_memory_utilization, max_num_seqs, 
                       enable_prefix_caching, enable_chunked_prefill, swap_space, 
                       max_num_batched_tokens, mps_percentage)
            )
        ''')
        
        self.conn.commit()
    
    def insert_result(self, result: ProfilingResult):
        """Insert a profiling result into database"""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO profiling_results (
                    power_cap, gpu_memory_utilization, max_num_seqs,
                    enable_prefix_caching, enable_chunked_prefill,
                    swap_space, max_num_batched_tokens, mps_percentage,
                    prompt_id, prompt, prompt_length,
                    ttft, tbt, total_latency, tokens_generated, throughput,
                    energy_kwh, avg_power_kw, peak_power_kw,
                    gpu_utilization, gpu_memory_used_gb,
                    timestamp, generated_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.power_cap, result.gpu_memory_utilization, result.max_num_seqs,
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
    
    def close(self):
        """Close database connection"""
        self.conn.close()

class DatasetLoader:
    """Load and manage prompt datasets"""
    
    def __init__(self, prompt_file: str, limit: int = None):
        self.prompt_file = prompt_file
        self.prompts = []
        self.limit = limit
        self.load_data()
    
    def load_data(self):
        """Load prompt data from JSONL file"""
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
            # Generate mock prompts for testing
            print(f"Warning: Prompt file not found: {self.prompt_file}")
            print("Generating mock prompts for testing...")
            self.prompts = [
                f"What is the capital of country {i}?" for i in range(10)
            ]
            print(f"Generated {len(self.prompts)} mock prompts")
        
        if not self.prompts:
            raise ValueError("No prompts available")
    
    def get_all_prompts(self) -> List[str]:
        """Get all prompts"""
        return self.prompts

class PowerMonitor:
    """Monitor GPU power consumption (mock or real)"""
    
    def __init__(self, gpu_index=0, mock=False):
        self.gpu_index = gpu_index
        self.mock = mock
        if not mock:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        else:
            self.handle = f"mock_gpu_{gpu_index}"
        self.power_readings = []
        self.monitoring = False
        self.thread = None
        
    def start(self):
        """Start power monitoring"""
        self.power_readings = []
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()
        
    def stop(self):
        """Stop power monitoring and return results"""
        self.monitoring = False
        if self.thread:
            self.thread.join()
        return self.power_readings
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                if self.mock or MOCK_MODE:
                    # Generate mock data
                    power = random.uniform(100, 250) * 1000  # Convert to watts
                    gpu_util = random.uniform(50, 100)
                    mem_used = random.uniform(4, 12) * (1024**3)
                else:
                    power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert to watts
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    gpu_util = utilization.gpu
                    mem_used = mem_info.used
                
                self.power_readings.append({
                    'timestamp': time.time(),
                    'power_w': power / 1000.0 if self.mock or MOCK_MODE else power,
                    'gpu_utilization': gpu_util,
                    'memory_used_gb': mem_used / (1024**3)
                })
            except Exception as e:
                print(f"Error reading power: {e}")
            time.sleep(0.1)  # Sample every 100ms

class VLLMProfiler:
    """Simplified profiler class for vLLM inference"""
    
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", port=8000, 
                 prompt_file="../dataset/grad_school_math/train.jsonl", output_dir=None, prompt_limit=None):
        self.model_name = model_name
        self.port = port
        self.server_process = None
        self.power_monitor = PowerMonitor(mock=MOCK_MODE)
        
        # Load prompts
        self.dataset_loader = DatasetLoader(prompt_file, limit=prompt_limit)
        
        # Create output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"vllm_profiling_{timestamp}"
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        
        # Initialize database
        db_path = os.path.join(self.output_dir, "profiling_results.db")
        self.db_manager = DatabaseManager(db_path)
        print(f"Database created at: {db_path}")
        
        # Define parameter ranges for experimentation
        # Test
        # self.param_ranges = {
        #     'power_cap': [150, 200],
        #     'gpu_memory_utilization': [0.9],
        #     'max_num_seqs': [64, 128],
        #     'enable_prefix_caching': [True],
        #     'enable_chunked_prefill': [False],
        #     'swap_space': [4],
        #     'max_num_batched_tokens': [2048],
        #     'mps_percentage': [100]
        # }
        # V100S, 4*4*2*2*2*3 = 384 configs
        self.param_ranges = {
            'power_cap': [150, 200, 250],  # limit 250
            'gpu_memory_utilization': [0.90, 0.95],  # 32GB 
            'max_num_seqs': [64, 128], # batch size
            'enable_prefix_caching': [True],
            'enable_chunked_prefill': [False],  
            'swap_space': [4, 8],
            'max_num_batched_tokens': [2048, 4096], # 批处理的最大token数
            'mps_percentage': [50, 75, 100]
        }
        
        # Save configuration
        config_file = os.path.join(self.output_dir, "experiment_config.json")
        with open(config_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'port': port,
                'prompt_file': prompt_file,
                'param_ranges': self.param_ranges,
                'mock_mode': MOCK_MODE,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
    
    def set_power_cap(self, power_cap: int, gpu_index: int = 0):
        """Set GPU power cap (mock or real)"""
        if MOCK_MODE:
            print(f"Mock: Power cap set to {power_cap}W")
        else:
            cmd = f"sudo nvidia-smi -i {gpu_index} -pl {power_cap}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: Failed to set power cap: {result.stderr}")
            else:
                print(f"Power cap set to {power_cap}W")
            
    def set_mps_percentage(self, percentage: int):
        """Set CUDA MPS active thread percentage"""
        os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(percentage)
        print(f"MPS active thread percentage set to {percentage}%")
    
    def check_server_health(self, max_retries: int = 60, retry_delay: int = 2):
        """Check if vLLM server is ready"""
        url = f"http://localhost:{self.port}/health"
        
        for i in range(max_retries):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"Server is ready! (took {i * retry_delay} seconds)")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            if self.server_process and self.server_process.poll() is not None:
                stdout, stderr = self.server_process.communicate()
                print(f"Server failed to start. Check logs for details.")
                return False
            
            if i % 10 == 0 and i > 0:
                print(f"Still waiting for server... ({i * retry_delay} seconds elapsed)")
            
            time.sleep(retry_delay)
        
        print(f"Server failed to start after {max_retries * retry_delay} seconds")
        return False
    
    def start_vllm_server(self, config: Dict[str, Any]):
        """Start vLLM server (mock or real)"""
        if MOCK_MODE:
            print(f"Mock: Starting vLLM server with config: {config}")
            return
        
        self.stop_vllm_server()
        
        # Check port availability
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', self.port))
        sock.close()
        if result == 0:
            print(f"Port {self.port} is in use. Killing existing process...")
            subprocess.run(f"lsof -ti:{self.port} | xargs kill -9", shell=True)
            time.sleep(2)
        
        # Build command - NOTE: using --enable-log-requests instead of deprecated --disable-log-requests
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(config['gpu_memory_utilization']),
            "--max-num-seqs", str(config['max_num_seqs']),
            "--swap-space", str(config['swap_space']),
            "--max-num-batched-tokens", str(config['max_num_batched_tokens']),
            "--max-model-len", "512", # context length
            "--enforce-eager"
        ]
        
        if config['enable_prefix_caching']:
            cmd.append("--enable-prefix-caching")
        
        if config['enable_chunked_prefill']:
            cmd.append("--enable-chunked-prefill")
        
        print(f"Starting vLLM server with config: {config}")
        print(f"Command: {' '.join(cmd)}")
        
        log_file = os.path.join(self.output_dir, "logs", f"vllm_server_{time.time()}.log")
        with open(log_file, 'w') as f:
            self.server_process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        
        print(f"Server log file: {log_file}")
        
        # Wait for server to be ready
        if not self.check_server_health():
            # If server failed to start, read the log file for debugging
            with open(log_file, 'r') as f:
                print("\nServer startup failed. Last 50 lines of log:")
                lines = f.readlines()
                for line in lines[-50:]:
                    print(line.rstrip())
            raise RuntimeError("vLLM server failed to start")
    
    def stop_vllm_server(self):
        """Stop vLLM server"""
        if MOCK_MODE:
            print("Mock: Stopping vLLM server")
            return
            
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
        
        subprocess.run("pkill -f vllm.entrypoints.openai.api_server", shell=True)
        time.sleep(2)
    
    async def send_single_request(self, prompt: str, prompt_id: int) -> Dict[str, Any]:
        """Send a single inference request (mock or real)"""
        if MOCK_MODE:
            # Generate mock results
            await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate processing time
            
            return {
                'prompt_id': prompt_id,
                'prompt': prompt,
                'prompt_length': len(prompt.split()),
                'ttft': random.uniform(0.1, 0.5),
                'tbt': random.uniform(0.01, 0.05),
                'total_latency': random.uniform(1.0, 3.0),
                'tokens_generated': random.randint(50, 150),
                'generated_text': f"Mock response for prompt {prompt_id}",
                'inter_token_times': [random.uniform(0.01, 0.05) for _ in range(10)]
            }
        
        # Real inference request
        url = f"http://localhost:{self.port}/v1/completions"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": True
        }
        
        result = {
            'prompt_id': prompt_id,
            'prompt': prompt,
            'prompt_length': len(prompt.split()),
            'ttft': 0,
            'tbt': 0,
            'total_latency': 0,
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
                
                if result['inter_token_times']:
                    result['tbt'] = np.mean(result['inter_token_times'])
                
            except Exception as e:
                print(f"Error during inference for prompt {prompt_id}: {e}")
        
        return result
    
    def calculate_energy_metrics(self, power_readings: List[Dict]) -> Dict[str, float]:
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
    
    def run_config_experiment(self, config: Dict[str, Any]) -> List[ProfilingResult]:
        """Run experiments for a single configuration with all prompts"""
        print(f"\n{'='*60}")
        print(f"Running experiments with configuration:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}")
        
        # Set hardware configurations
        self.set_power_cap(config['power_cap'])
        self.set_mps_percentage(config['mps_percentage'])
        
        # Start vLLM server
        self.start_vllm_server(config)
        
        # Get all prompts
        prompts = self.dataset_loader.get_all_prompts()
        results = []
        
        print(f"Processing {len(prompts)} prompts...")
        
        # Process each prompt
        for prompt_id, prompt in enumerate(prompts):
            print(f"  Processing prompt {prompt_id + 1}/{len(prompts)}...")
            
            # Start power monitoring
            self.power_monitor.start()
            
            # Run inference
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            inference_result = loop.run_until_complete(
                self.send_single_request(prompt=prompt, prompt_id=prompt_id)
            )
            loop.close()
            
            # Stop power monitoring
            power_readings = self.power_monitor.stop()
            energy_metrics = self.calculate_energy_metrics(power_readings)
            
            # Calculate throughput
            throughput = (inference_result['tokens_generated'] / 
                         inference_result['total_latency'] 
                         if inference_result['total_latency'] > 0 else 0)
            
            # Create result object
            result = ProfilingResult(
                # Configuration
                power_cap=config['power_cap'],
                gpu_memory_utilization=config['gpu_memory_utilization'],
                max_num_seqs=config['max_num_seqs'],
                enable_prefix_caching=config['enable_prefix_caching'],
                enable_chunked_prefill=config['enable_chunked_prefill'],
                swap_space=config['swap_space'],
                max_num_batched_tokens=config['max_num_batched_tokens'],
                mps_percentage=config['mps_percentage'],
                
                # Request info
                prompt_id=inference_result['prompt_id'],
                prompt=inference_result['prompt'],
                prompt_length=inference_result['prompt_length'],
                
                # Performance metrics
                ttft=inference_result['ttft'],
                tbt=inference_result['tbt'],
                total_latency=inference_result['total_latency'],
                tokens_generated=inference_result['tokens_generated'],
                throughput=throughput,
                
                # Energy metrics
                energy_kwh=energy_metrics['energy_kwh'],
                avg_power_kw=energy_metrics['avg_power_kw'],
                peak_power_kw=energy_metrics['peak_power_kw'],
                
                # System metrics
                gpu_utilization=energy_metrics['avg_gpu_utilization'],
                gpu_memory_used_gb=energy_metrics['avg_memory_used_gb'],
                
                # Metadata
                timestamp=datetime.now().isoformat(),
                generated_text=inference_result['generated_text']
            )
            
            # Save to database
            self.db_manager.insert_result(result)
            results.append(result)
        
        # Stop server
        self.stop_vllm_server()
        
        print(f"Completed {len(results)} prompts for this configuration")
        return results
    
    def run_all_experiments(self):
        """Run experiments for all parameter combinations"""
        # Generate all configurations
        all_configs = list(itertools.product(*[
            self.param_ranges['power_cap'],
            self.param_ranges['gpu_memory_utilization'],
            self.param_ranges['max_num_seqs'],
            self.param_ranges['enable_prefix_caching'],
            self.param_ranges['enable_chunked_prefill'],
            self.param_ranges['swap_space'],
            self.param_ranges['max_num_batched_tokens'],
            self.param_ranges['mps_percentage']
        ]))
        
        print(f"Total configurations to test: {len(all_configs)}")
        
        all_results = []
        
        for config_idx, config_tuple in enumerate(all_configs, 1):
            config = {
                'power_cap': config_tuple[0],
                'gpu_memory_utilization': config_tuple[1],
                'max_num_seqs': config_tuple[2],
                'enable_prefix_caching': config_tuple[3],
                'enable_chunked_prefill': config_tuple[4],
                'swap_space': config_tuple[5],
                'max_num_batched_tokens': config_tuple[6],
                'mps_percentage': config_tuple[7]
            }
            
            print(f"\n{'='*70}")
            print(f"Configuration {config_idx}/{len(all_configs)}")
            print(f"{'='*70}")
            
            try:
                results = self.run_config_experiment(config)
                all_results.extend(results)
                
                # Export current results to CSV
                self.export_results_to_csv()
                
            except Exception as e:
                print(f"Error in configuration {config_idx}: {e}")
                continue
            
            # Cool down between configurations
            if not MOCK_MODE:
                print("Cooling down before next configuration...")
                time.sleep(10)
        
        return all_results
    
    def export_results_to_csv(self):
        """Export database results to CSV files"""
        conn = sqlite3.connect(os.path.join(self.output_dir, "profiling_results.db"))
        
        # Export detailed results
        df_results = pd.read_sql_query("SELECT * FROM profiling_results", conn)
        csv_path = os.path.join(self.output_dir, "detailed_results.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"Detailed results exported to: {csv_path}")
        
        conn.close()


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified vLLM Profiling Script')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                       help='Model name to profile')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port for vLLM server')
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
    
    # Initialize profiler
    profiler = VLLMProfiler(
        model_name=args.model,
        port=args.port,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir,
        prompt_limit=args.prompt_limit
    )
    
    try:
        # Run all experiments
        results = profiler.run_all_experiments()
        
        print(f"\n{'='*70}")
        print("PROFILING COMPLETE")
        print(f"{'='*70}")
        print(f"Total results collected: {len(results)}")
        print(f"Results saved to: {profiler.output_dir}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        profiler.stop_vllm_server()
        profiler.db_manager.close()
        if not MOCK_MODE:
            pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()