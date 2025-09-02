#!/usr/bin/env python3
"""
vLLM Inference Profiling Script with Real Dataset
Uses trace.csv and train.jsonl for realistic profiling
"""

import os
import sys
import time
import json
import subprocess
import itertools
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import asyncio
import aiohttp
from dataclasses import dataclass, asdict
import pynvml
import threading
import queue
import requests
import sqlite3
import random
from typing import Dict, List, Any, Optional

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
    instance_sn: str
    scheduled_time: str
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
        
        # Main profiling results table
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
                
                -- Request info
                instance_sn TEXT,
                scheduled_time TEXT,
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
                
                -- Indexes for common queries
                UNIQUE(instance_sn, power_cap, gpu_memory_utilization, max_num_seqs, 
                       enable_prefix_caching, enable_chunked_prefill, swap_space, 
                       max_num_batched_tokens, mps_percentage)
            )
        ''')
        
        # Configuration summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS config_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_hash TEXT UNIQUE,
                power_cap INTEGER,
                gpu_memory_utilization REAL,
                max_num_seqs INTEGER,
                enable_prefix_caching BOOLEAN,
                enable_chunked_prefill BOOLEAN,
                swap_space INTEGER,
                max_num_batched_tokens INTEGER,
                mps_percentage INTEGER,
                num_requests INTEGER,
                avg_ttft REAL,
                avg_tbt REAL,
                avg_throughput REAL,
                avg_energy_kwh REAL,
                timestamp TEXT
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
                    instance_sn, scheduled_time, prompt, prompt_length,
                    ttft, tbt, total_latency, tokens_generated, throughput,
                    energy_kwh, avg_power_kw, peak_power_kw,
                    gpu_utilization, gpu_memory_used_gb,
                    timestamp, generated_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.power_cap, result.gpu_memory_utilization, result.max_num_seqs,
                result.enable_prefix_caching, result.enable_chunked_prefill,
                result.swap_space, result.max_num_batched_tokens, result.mps_percentage,
                result.instance_sn, result.scheduled_time, result.prompt, result.prompt_length,
                result.ttft, result.tbt, result.total_latency, result.tokens_generated, result.throughput,
                result.energy_kwh, result.avg_power_kw, result.peak_power_kw,
                result.gpu_utilization, result.gpu_memory_used_gb,
                result.timestamp, result.generated_text
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            self.conn.rollback()
    
    def update_config_summary(self, config: Dict[str, Any], results: List[ProfilingResult]):
        """Update configuration summary statistics"""
        if not results:
            return
            
        cursor = self.conn.cursor()
        
        # Calculate summary statistics
        avg_ttft = np.mean([r.ttft for r in results])
        avg_tbt = np.mean([r.tbt for r in results])
        avg_throughput = np.mean([r.throughput for r in results])
        avg_energy = np.mean([r.energy_kwh for r in results])
        
        # Create config hash for unique identification
        config_str = json.dumps(config, sort_keys=True)
        config_hash = str(hash(config_str))
        
        cursor.execute('''
            INSERT OR REPLACE INTO config_summary (
                config_hash, power_cap, gpu_memory_utilization, max_num_seqs,
                enable_prefix_caching, enable_chunked_prefill,
                swap_space, max_num_batched_tokens, mps_percentage,
                num_requests, avg_ttft, avg_tbt, avg_throughput, avg_energy_kwh,
                timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            config_hash,
            config['power_cap'], config['gpu_memory_utilization'], config['max_num_seqs'],
            config['enable_prefix_caching'], config['enable_chunked_prefill'],
            config['swap_space'], config['max_num_batched_tokens'], config['mps_percentage'],
            len(results), avg_ttft, avg_tbt, avg_throughput, avg_energy,
            datetime.now().isoformat()
        ))
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        self.conn.close()

class DatasetLoader:
    """Load and manage trace and prompt datasets"""
    
    def __init__(self, trace_file: str, prompt_file: str):
        self.trace_file = trace_file
        self.prompt_file = prompt_file
        self.trace_data = None
        self.prompts = None
        self.load_data()
    
    def load_data(self):
        """Load trace and prompt data"""
        # Load trace data
        if os.path.exists(self.trace_file):
            self.trace_data = pd.read_csv(self.trace_file)
            print(f"Loaded {len(self.trace_data)} trace records from {self.trace_file}")
        else:
            raise FileNotFoundError(f"Trace file not found: {self.trace_file}")
        
        # Load prompts from JSONL
        self.prompts = []
        if os.path.exists(self.prompt_file):
            with open(self.prompt_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'question' in data:
                            self.prompts.append(data['question'])
                    except json.JSONDecodeError:
                        continue
            print(f"Loaded {len(self.prompts)} prompts from {self.prompt_file}")
        else:
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file}")
        
        if not self.prompts:
            raise ValueError("No prompts found in the JSONL file")
    
    def get_random_prompt(self) -> str:
        """Get a random prompt from the dataset"""
        return random.choice(self.prompts)
    
    def get_trace_records_with_timing(self, limit: Optional[int] = None) -> List[Dict]:
        """Get trace records with timing information preserved"""
        records = []
        df = self.trace_data.head(limit) if limit else self.trace_data
        
        # Sort by creation_time to ensure proper order
        df = df.sort_values('creation_time')
        
        for _, row in df.iterrows():
            records.append({
                'instance_sn': row['instance_sn'],
                'creation_time': row['creation_time']
            })
        
        return records

class PowerMonitor:
    """Monitor GPU power consumption in a separate thread"""
    
    def __init__(self, gpu_index=0):
        self.gpu_index = gpu_index
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
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
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert to watts
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                
                self.power_readings.append({
                    'timestamp': time.time(),
                    'power_w': power,
                    'gpu_utilization': utilization.gpu,
                    'memory_used_gb': mem_info.used / (1024**3)
                })
            except Exception as e:
                print(f"Error reading power: {e}")
            time.sleep(0.1)  # Sample every 100ms

class VLLMProfiler:
    """Main profiler class for vLLM inference with real datasets"""
    
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", port=8000, 
                 trace_file="trace.csv", prompt_file="train.jsonl",
                 output_dir=None):
        self.model_name = model_name
        self.port = port
        self.server_process = None
        self.power_monitor = PowerMonitor()
        
        # Load datasets
        self.dataset_loader = DatasetLoader(trace_file, prompt_file)
        
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
        self.param_ranges = {
            'power_cap': [100, 150, 200],
            'gpu_memory_utilization': [0.9],
            'max_num_seqs': [64, 128],
            'enable_prefix_caching': [True],
            'enable_chunked_prefill': [False],
            'swap_space': [4],
            'max_num_batched_tokens': [2048, 4096],
            'mps_percentage': [75, 100]
        }
        
        # Save configuration
        config_file = os.path.join(self.output_dir, "experiment_config.json")
        with open(config_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'port': port,
                'trace_file': trace_file,
                'prompt_file': prompt_file,
                'param_ranges': self.param_ranges,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
    
    def set_power_cap(self, power_cap: int, gpu_index: int = 0):
        """Set GPU power cap using nvidia-smi"""
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
        """Start vLLM server with specified configuration"""
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
        
        # Build command
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(config['gpu_memory_utilization']),
            "--max-num-seqs", str(config['max_num_seqs']),
            "--swap-space", str(config['swap_space']),
            "--max-num-batched-tokens", str(config['max_num_batched_tokens']),
            "--disable-log-requests",
            "--max-model-len", "512",
            "--enforce-eager"
        ]
        
        if config['enable_prefix_caching']:
            cmd.append("--enable-prefix-caching")
        
        if config['enable_chunked_prefill']:
            cmd.append("--enable-chunked-prefill")
        
        print(f"Starting vLLM server with config: {config}")
        
        log_file = os.path.join(self.output_dir, "logs", f"vllm_server_{time.time()}.log")
        with open(log_file, 'w') as f:
            self.server_process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        
        if not self.check_server_health():
            raise RuntimeError("vLLM server failed to start")
    
    def stop_vllm_server(self):
        """Stop vLLM server"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
        
        subprocess.run("pkill -f vllm.entrypoints.openai.api_server", shell=True)
        time.sleep(2)
    
    async def send_single_request(self, prompt: str, instance_sn: str, 
                                 scheduled_time: str) -> Dict[str, Any]:
        """Send a single inference request and collect metrics"""
        url = f"http://localhost:{self.port}/v1/completions"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 100, # TBD
            "temperature": 0.7,
            "stream": True
        }
        
        result = {
            'instance_sn': instance_sn,
            'scheduled_time': scheduled_time,
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
                                elif data_str == '[DONE]':
                                    break
                
                result['total_latency'] = time.time() - start_time
                
                if result['inter_token_times']:
                    result['tbt'] = np.mean(result['inter_token_times'])
                
            except Exception as e:
                print(f"Error during inference for {instance_sn}: {e}")
        
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
    
    def run_config_experiment(self, config: Dict[str, Any], 
                            trace_limit: Optional[int] = None,
                            time_scale: float = 1.0) -> List[ProfilingResult]:
        """
        Run experiments for a single configuration with all trace records
        
        Args:
            config: Configuration parameters
            trace_limit: Limit number of trace records to process
            time_scale: Scale factor for time intervals (e.g., 0.1 for 10x speedup)
        """
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
        
        # Get trace records with timing information
        trace_records = self.dataset_loader.get_trace_records_with_timing(limit=trace_limit)
        results = []
        
        print(f"Processing {len(trace_records)} trace records...")
        print(f"Time scale: {time_scale} (1.0 = real-time, 0.1 = 10x faster)")
        
        # Start time reference
        experiment_start_time = time.time()
        # first_trace_time = None
        
        # Process each trace record with proper timing
        for i, trace in enumerate(trace_records, 1):
            # print("before running everything", i, trace)

            # creation_time is already in seconds from experiment start
            trace_time_seconds = float(trace['creation_time'])
            
            # Apply time scaling to get actual wait time
            scaled_time_offset = trace_time_seconds * time_scale
            
            # Calculate when we should send this request
            target_time = experiment_start_time + scaled_time_offset
            current_time = time.time()

            # print("trace_time_seconds =", trace_time_seconds, "scaled_time_offset =", scaled_time_offset)
            # print("target time =", target_time, "current time =", current_time)
            
            
            # Wait until it's time to send this request
            if target_time > current_time:
                wait_time = target_time - current_time
                if wait_time > 0.1:  # Only print if waiting more than 100ms
                    print(f"  Request {i}/{len(trace_records)}: Waiting {wait_time:.2f}s (trace time: {trace_time_seconds:.2f}s)...")
                time.sleep(wait_time)
            else:
                # We're behind schedule
                delay = current_time - target_time
                if delay > 1.0:  # Only warn if significantly behind
                    print(f"  Request {i}/{len(trace_records)}: Running {delay:.2f}s behind schedule")
            

            # Get random prompt
            prompt = self.dataset_loader.get_random_prompt()
            
            # Start power monitoring
            self.power_monitor.start()
            
            # Record actual send time
            actual_send_time = time.time()
            
            # Run inference
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            inference_result = loop.run_until_complete(
                self.send_single_request(
                    prompt=prompt,
                    instance_sn=trace['instance_sn'],
                    scheduled_time=trace['creation_time']
                )
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
                instance_sn=inference_result['instance_sn'],
                scheduled_time=inference_result['scheduled_time'],
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
            
            # Print progress
            if i % 10 == 0 or i == len(trace_records):
                elapsed = time.time() - experiment_start_time
                print(f"  Processed {i}/{len(trace_records)} requests in {elapsed:.1f}s")
        
        # Update configuration summary
        self.db_manager.update_config_summary(config, results)
        
        # Stop server
        self.stop_vllm_server()
        
        total_time = time.time() - experiment_start_time
        print(f"Completed {len(results)} requests in {total_time:.1f}s for this configuration")
        return results
    
    def run_all_experiments(self, trace_limit: Optional[int] = None, time_scale: float = 1.0):
        """
        Run experiments for all parameter combinations
        
        Args:
            trace_limit: Limit number of trace records to process
            time_scale: Scale factor for time intervals (e.g., 0.1 for 10x speedup)
        """
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
        print(f"Time scale: {time_scale} (1.0 = real-time replay)")
        
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
                results = self.run_config_experiment(config, trace_limit, time_scale)
                all_results.extend(results)
                
                # Export current results to CSV
                self.export_results_to_csv()
                
            except Exception as e:
                print(f"Error in configuration {config_idx}: {e}")
                continue
            
            # Cool down between configurations
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
        
        # Export configuration summary
        df_summary = pd.read_sql_query("SELECT * FROM config_summary", conn)
        summary_path = os.path.join(self.output_dir, "config_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"Configuration summary exported to: {summary_path}")
        
        conn.close()
        
    def generate_analysis_report(self):
        """Generate analysis report from database"""
        conn = sqlite3.connect(os.path.join(self.output_dir, "profiling_results.db"))
        
        # Load data
        df = pd.read_sql_query("SELECT * FROM profiling_results", conn)
        df_summary = pd.read_sql_query("SELECT * FROM config_summary", conn)
        
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("VLLM PROFILING ANALYSIS REPORT")
        report_lines.append(f"Generated at: {datetime.now().isoformat()}")
        report_lines.append("="*70)
        
        # Overall statistics
        report_lines.append("\nOVERALL STATISTICS:")
        report_lines.append(f"  Total experiments: {len(df)}")
        report_lines.append(f"  Total configurations tested: {len(df_summary)}")
        report_lines.append(f"  Average TTFT: {df['ttft'].mean():.4f} seconds")
        report_lines.append(f"  Average TBT: {df['tbt'].mean():.4f} seconds")
        report_lines.append(f"  Average throughput: {df['throughput'].mean():.2f} tokens/second")
        report_lines.append(f"  Average energy: {df['energy_kwh'].mean():.6f} kWh")
        
        # Best configurations for each metric
        report_lines.append("\n" + "="*70)
        report_lines.append("BEST CONFIGURATIONS BY METRIC:")
        report_lines.append("="*70)
        
        # Best for latency (TTFT)
        best_ttft_config = df_summary.nsmallest(1, 'avg_ttft').iloc[0]
        report_lines.append("\nBest for TTFT (lowest):")
        report_lines.append(f"  Average TTFT: {best_ttft_config['avg_ttft']:.4f} seconds")
        report_lines.append(f"  Configuration:")
        report_lines.append(f"    Power cap: {best_ttft_config['power_cap']}W")
        report_lines.append(f"    GPU memory: {best_ttft_config['gpu_memory_utilization']}")
        report_lines.append(f"    Max sequences: {best_ttft_config['max_num_seqs']}")
        report_lines.append(f"    MPS percentage: {best_ttft_config['mps_percentage']}%")
        
        # Best for throughput
        best_throughput_config = df_summary.nlargest(1, 'avg_throughput').iloc[0]
        report_lines.append("\nBest for Throughput (highest):")
        report_lines.append(f"  Average throughput: {best_throughput_config['avg_throughput']:.2f} tokens/s")
        report_lines.append(f"  Configuration:")
        report_lines.append(f"    Power cap: {best_throughput_config['power_cap']}W")
        report_lines.append(f"    GPU memory: {best_throughput_config['gpu_memory_utilization']}")
        report_lines.append(f"    Max sequences: {best_throughput_config['max_num_seqs']}")
        report_lines.append(f"    MPS percentage: {best_throughput_config['mps_percentage']}%")
        
        # Best for energy efficiency
        df_summary['efficiency'] = df_summary['avg_throughput'] / (df_summary['avg_energy_kwh'] * 1000)
        best_efficiency_config = df_summary.nlargest(1, 'efficiency').iloc[0]
        report_lines.append("\nBest for Energy Efficiency (throughput/energy):")
        report_lines.append(f"  Efficiency: {best_efficiency_config['efficiency']:.2f} tokens/Wh")
        report_lines.append(f"  Configuration:")
        report_lines.append(f"    Power cap: {best_efficiency_config['power_cap']}W")
        report_lines.append(f"    GPU memory: {best_efficiency_config['gpu_memory_utilization']}")
        report_lines.append(f"    Max sequences: {best_efficiency_config['max_num_seqs']}")
        report_lines.append(f"    MPS percentage: {best_efficiency_config['mps_percentage']}%")
        
        # Parameter impact analysis
        report_lines.append("\n" + "="*70)
        report_lines.append("PARAMETER IMPACT ANALYSIS:")
        report_lines.append("="*70)
        
        # Analyze impact of each parameter
        params_to_analyze = ['power_cap', 'gpu_memory_utilization', 'max_num_seqs', 
                            'mps_percentage', 'enable_prefix_caching']
        
        for param in params_to_analyze:
            if param in df.columns:
                report_lines.append(f"\nImpact of {param}:")
                grouped = df.groupby(param).agg({
                    'ttft': 'mean',
                    'throughput': 'mean',
                    'energy_kwh': 'mean'
                }).round(4)
                
                for idx, row in grouped.iterrows():
                    report_lines.append(f"  {param}={idx}:")
                    report_lines.append(f"    Avg TTFT: {row['ttft']:.4f}s")
                    report_lines.append(f"    Avg Throughput: {row['throughput']:.2f} tokens/s")
                    report_lines.append(f"    Avg Energy: {row['energy_kwh']:.6f} kWh")
        
        # Top 5 most efficient configurations
        report_lines.append("\n" + "="*70)
        report_lines.append("TOP 5 MOST EFFICIENT CONFIGURATIONS:")
        report_lines.append("="*70)
        
        top_5_efficient = df_summary.nlargest(5, 'efficiency')
        for idx, (_, config) in enumerate(top_5_efficient.iterrows(), 1):
            report_lines.append(f"\n{idx}. Efficiency: {config['efficiency']:.2f} tokens/Wh")
            report_lines.append(f"   Throughput: {config['avg_throughput']:.2f} tokens/s")
            report_lines.append(f"   Energy: {config['avg_energy_kwh']:.6f} kWh")
            report_lines.append(f"   TTFT: {config['avg_ttft']:.4f}s")
            report_lines.append(f"   Config: power={config['power_cap']}W, gpu_mem={config['gpu_memory_utilization']}, "
                              f"max_seqs={config['max_num_seqs']}, mps={config['mps_percentage']}%")
        
        # Save report
        report_text = '\n'.join(report_lines)
        report_path = os.path.join(self.output_dir, "analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport saved to: {report_path}")
        
        conn.close()