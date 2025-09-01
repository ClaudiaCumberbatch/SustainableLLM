#!/usr/bin/env python3
"""
vLLM Inference Profiling Script for Llama2
Collects performance metrics under different parameter configurations
"""

import os
import sys
import time
import json
import subprocess
import itertools
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import asyncio
import aiohttp
from dataclasses import dataclass, asdict
import pynvml
import threading
import queue
import requests  # Add requests for health check

# Initialize NVIDIA Management Library
pynvml.nvmlInit()

@dataclass
class ProfilingResult:
    """Data class for storing profiling results"""
    power_cap: int
    gpu_memory_utilization: float
    max_num_seqs: int
    enable_prefix_caching: bool
    enable_chunked_prefill: bool
    swap_space: int
    max_num_batched_tokens: int
    mps_percentage: int
    ttft: float  # Time to first token
    tbt: float   # Time between tokens
    throughput: float  # Tokens per second
    energy_kwh: float  # Energy consumption in kWh
    avg_power_kw: float  # Average power in kW
    timestamp: str

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
                self.power_readings.append({
                    'timestamp': time.time(),
                    'power_w': power
                })
            except Exception as e:
                print(f"Error reading power: {e}")
            time.sleep(0.1)  # Sample every 100ms

class VLLMProfiler:
    """Main profiler class for vLLM inference"""
    
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", port=8000, tensor_parallel_size=1, output_dir=None):
        self.model_name = model_name
        self.port = port
        self.tensor_parallel_size = tensor_parallel_size
        self.server_process = None
        self.power_monitor = PowerMonitor()
        
        # Create output directory with timestamp
        if output_dir:
            self.output_dir = output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"vllm_profiling_{timestamp}"
        
        # Create directory structure
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        
        # File paths
        self.results_csv = os.path.join(self.output_dir, "profiling_results.csv")
        self.summary_file = os.path.join(self.output_dir, "summary_report.txt")
        self.server_stdout_log = os.path.join(self.output_dir, "logs", "vllm_server_stdout.log")
        self.server_stderr_log = os.path.join(self.output_dir, "logs", "vllm_server_stderr.log")
        
        print(f"Output directory: {self.output_dir}")
        
        # Save configuration
        config_file = os.path.join(self.output_dir, "experiment_config.json")
        with open(config_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'port': port,
                'tensor_parallel_size': tensor_parallel_size,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Define parameter ranges for experimentation
        # !!! Change here
        # self.param_ranges = {
        #     'power_cap': [100, 150, 200, 250, 300],  # Watts
        #     'gpu_memory_utilization': [0.5, 0.7, 0.9, 0.95],
        #     'max_num_seqs': [64, 128, 256],
        #     'enable_prefix_caching': [True, False],
        #     'enable_chunked_prefill': [True, False],
        #     'swap_space': [0, 2, 4, 8],  # GB
        #     'max_num_batched_tokens': [2048, 4096, 8192],
        #     'mps_percentage': [50, 75, 100]  # MPS active thread percentage
        # }

        self.param_ranges = {
            'power_cap': [100], # good
            'gpu_memory_utilization': [0.9], # only these two
            'max_num_seqs': [128], # good
            'enable_prefix_caching': [True], # only
            'enable_chunked_prefill': [False], # only
            'swap_space': [4],  # GB  good
            'max_num_batched_tokens': [4096], # good
            'mps_percentage': [50, 75, 100] # good
        }
        
        # Test prompts for inference
        self.test_prompts = [
            "Explain the concept of machine learning in simple terms.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "How does the internet work?",
            "What is quantum computing?"
        ]
        
        # Save parameter ranges to file for reference
        param_file = os.path.join(self.output_dir, "parameter_ranges.json")
        with open(param_file, 'w') as f:
            json.dump(self.param_ranges, f, indent=2)
        
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
        import requests
        
        url = f"http://localhost:{self.port}/health"
        
        for i in range(max_retries):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"Server is ready! (took {i * retry_delay} seconds)")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            # Check if process is still running
            if self.server_process and self.server_process.poll() is not None:
                # Process has terminated
                stdout, stderr = self.server_process.communicate()
                print(f"Server failed to start. Error output:")
                print(stderr.decode('utf-8'))
                return False
            
            if i % 10 == 0 and i > 0:
                print(f"Still waiting for server... ({i * retry_delay} seconds elapsed)")
            
            time.sleep(retry_delay)
        
        print(f"Server failed to start after {max_retries * retry_delay} seconds")
        return False
    
    def start_vllm_server(self, config: Dict[str, Any]):
        """Start vLLM server with specified configuration"""
        # Kill any existing server
        self.stop_vllm_server()
        
        # Check if port is available
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', self.port))
        sock.close()
        if result == 0:
            print(f"Warning: Port {self.port} is already in use. Attempting to kill existing process...")
            subprocess.run(f"lsof -ti:{self.port} | xargs kill -9", shell=True)
            time.sleep(2)
        
        # Build command
        # 这个地方从config里面拿参数写进命令里，所以直接修改这里可以直接改变实际运行的命令。
        # 如果只修改config，原本这里没有的项不会被运行到命令里。
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(config['gpu_memory_utilization']),
            "--max-num-seqs", str(config['max_num_seqs']),
            "--swap-space", str(config['swap_space']),
            "--max-num-batched-tokens", str(config['max_num_batched_tokens']),
            "--disable-log-requests",  # Reduce log verbosity
            "--max-model-len", "512",
            "--enforce-eager"
        ]
        
        if config['enable_prefix_caching']:
            cmd.append("--enable-prefix-caching")
        
        if config['enable_chunked_prefill']:
            cmd.append("--enable-chunked-prefill")
        
        # Start server with output redirection
        print(f"Starting vLLM server with config: {config}")
        print(f"Command: {' '.join(cmd)}")
        
        # Open log files with append mode for current experiment
        with open(self.server_stdout_log, 'a') as stdout_file, \
             open(self.server_stderr_log, 'a') as stderr_file:
            
            # Write experiment separator
            stdout_file.write(f"\n{'='*60}\n")
            stdout_file.write(f"Experiment started at {datetime.now().isoformat()}\n")
            stdout_file.write(f"Config: {json.dumps(config, indent=2)}\n")
            stdout_file.write(f"{'='*60}\n")
            
            stderr_file.write(f"\n{'='*60}\n")
            stderr_file.write(f"Experiment started at {datetime.now().isoformat()}\n")
            stderr_file.write(f"Config: {json.dumps(config, indent=2)}\n")
            stderr_file.write(f"{'='*60}\n")
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=stdout_file,
                stderr=stderr_file
            )
        
        print("Waiting for server to start...")
        
        # Check if server is ready
        if not self.check_server_health():
            print(f"Failed to start vLLM server. Check {self.server_stderr_log} for details.")
            self.stop_vllm_server()
            raise RuntimeError("vLLM server failed to start")
        
    def stop_vllm_server(self):
        """Stop vLLM server"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
        
        # Also kill any orphaned processes
        subprocess.run("pkill -f vllm.entrypoints.openai.api_server", shell=True)
        time.sleep(2)
        
    async def run_inference_test(self, prompts: List[str]) -> Dict[str, float]:
        """Run inference test and collect metrics"""
        url = f"http://localhost:{self.port}/v1/completions"
        
        metrics = {
            'ttft_list': [],
            'tbt_list': [],
            'total_tokens': 0,
            'total_time': 0
        }
        
        # Create inference results file
        inference_results_file = os.path.join(self.output_dir, "inference_results.jsonl")
        
        async with aiohttp.ClientSession() as session:
            for prompt_idx, prompt in enumerate(prompts):
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": 100,
                    "temperature": 0.7,
                    "stream": True
                }
                
                start_time = time.time()
                first_token_time = None
                prev_token_time = start_time
                token_count = 0
                inter_token_times = []
                generated_text = ""
                
                try:
                    async with session.post(url, json=payload) as response:
                        async for line in response.content:
                            if line:
                                line_str = line.decode('utf-8').strip()
                                if line_str.startswith('data: '):
                                    current_time = time.time()
                                    
                                    # Parse the SSE data
                                    data_str = line_str[6:]  # Remove 'data: ' prefix
                                    if data_str != '[DONE]':
                                        try:
                                            data = json.loads(data_str)
                                            if 'choices' in data and len(data['choices']) > 0:
                                                choice = data['choices'][0]
                                                if 'text' in choice:
                                                    generated_text += choice['text']
                                        except json.JSONDecodeError:
                                            pass
                                    
                                    if first_token_time is None:
                                        first_token_time = current_time
                                        metrics['ttft_list'].append(first_token_time - start_time)
                                    else:
                                        inter_token_times.append(current_time - prev_token_time)
                                    
                                    prev_token_time = current_time
                                    token_count += 1
                                    
                                    if data_str == '[DONE]':
                                        break
                        
                        # Save inference result
                        with open(inference_results_file, 'a') as f:
                            result_entry = {
                                'timestamp': datetime.now().isoformat(),
                                'prompt_idx': prompt_idx,
                                'prompt': prompt,
                                'generated_text': generated_text,
                                'ttft': first_token_time - start_time if first_token_time else 0,
                                'total_time': time.time() - start_time,
                                'token_count': token_count
                            }
                            f.write(json.dumps(result_entry) + '\n')
                
                except Exception as e:
                    print(f"Error during inference: {e}")
                    continue
                
                if inter_token_times:
                    metrics['tbt_list'].extend(inter_token_times)
                
                metrics['total_tokens'] += token_count
                metrics['total_time'] += (time.time() - start_time)
        
        # Calculate aggregate metrics
        result = {
            'ttft': np.mean(metrics['ttft_list']) if metrics['ttft_list'] else 0,
            'tbt': np.mean(metrics['tbt_list']) if metrics['tbt_list'] else 0,
            'throughput': metrics['total_tokens'] / metrics['total_time'] if metrics['total_time'] > 0 else 0
        }
        
        return result
    
    def calculate_energy_metrics(self, power_readings: List[Dict]) -> Dict[str, float]:
        """Calculate energy consumption from power readings"""
        if not power_readings:
            return {'energy_kwh': 0, 'avg_power_kw': 0}
        
        # Calculate average power
        avg_power_w = np.mean([r['power_w'] for r in power_readings])
        avg_power_kw = avg_power_w / 1000.0
        
        # Calculate energy (integrate power over time)
        energy_j = 0
        for i in range(1, len(power_readings)):
            dt = power_readings[i]['timestamp'] - power_readings[i-1]['timestamp']
            avg_p = (power_readings[i]['power_w'] + power_readings[i-1]['power_w']) / 2
            energy_j += avg_p * dt
        
        energy_kwh = energy_j / (3600 * 1000)  # Convert J to kWh
        
        return {
            'energy_kwh': energy_kwh,
            'avg_power_kw': avg_power_kw
        }
    
    def run_single_experiment(self, config: Dict[str, Any]) -> ProfilingResult:
        """Run a single experiment with given configuration"""
        print(f"\n{'='*60}")
        print(f"Running experiment with configuration:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}")
        
        # Log experiment start
        experiment_log = os.path.join(self.output_dir, "logs", "experiment_log.txt")
        with open(experiment_log, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Experiment started at {datetime.now().isoformat()}\n")
            f.write(f"Configuration:\n")
            for k, v in config.items():
                f.write(f"  {k}: {v}\n")
        
        # Set hardware configurations
        self.set_power_cap(config['power_cap'])
        self.set_mps_percentage(config['mps_percentage'])
        
        # Start vLLM server with configuration
        server_config = {
            'gpu_memory_utilization': config['gpu_memory_utilization'],
            'max_num_seqs': config['max_num_seqs'],
            'enable_prefix_caching': config['enable_prefix_caching'],
            'enable_chunked_prefill': config['enable_chunked_prefill'],
            'swap_space': config['swap_space'],
            'max_num_batched_tokens': config['max_num_batched_tokens']
        }
        self.start_vllm_server(server_config)
        
        # Start power monitoring
        self.power_monitor.start()
        
        # Run inference tests
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        inference_metrics = loop.run_until_complete(
            self.run_inference_test(self.test_prompts)
        )
        loop.close()
        
        # Stop power monitoring
        power_readings = self.power_monitor.stop()
        energy_metrics = self.calculate_energy_metrics(power_readings)
        
        # Stop server
        self.stop_vllm_server()
        
        # Combine results
        result = ProfilingResult(
            power_cap=config['power_cap'],
            gpu_memory_utilization=config['gpu_memory_utilization'],
            max_num_seqs=config['max_num_seqs'],
            enable_prefix_caching=config['enable_prefix_caching'],
            enable_chunked_prefill=config['enable_chunked_prefill'],
            swap_space=config['swap_space'],
            max_num_batched_tokens=config['max_num_batched_tokens'],
            mps_percentage=config['mps_percentage'],
            ttft=inference_metrics['ttft'],
            tbt=inference_metrics['tbt'],
            throughput=inference_metrics['throughput'],
            energy_kwh=energy_metrics['energy_kwh'],
            avg_power_kw=energy_metrics['avg_power_kw'],
            timestamp=datetime.now().isoformat()
        )
        
        # Log experiment completion
        with open(experiment_log, 'a') as f:
            f.write(f"Experiment completed at {datetime.now().isoformat()}\n")
            f.write(f"Results: TTFT={result.ttft:.3f}s, TBT={result.tbt:.3f}s, ")
            f.write(f"Throughput={result.throughput:.2f} tokens/s, ")
            f.write(f"Energy={result.energy_kwh:.6f} kWh, Power={result.avg_power_kw:.3f} kW\n")
        
        return result
    
    def run_experiments(self, num_samples: int = None):
        """Run multiple experiments with different configurations"""
        results = []
        
        # Generate all possible combinations or sample
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
        
        # Sample if requested
        if num_samples and num_samples < len(all_configs):
            import random
            all_configs = random.sample(all_configs, num_samples)
        
        print(f"Total experiments to run: {len(all_configs)}")
        
        for i, config_tuple in enumerate(all_configs, 1):
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
            
            print(f"\nExperiment {i}/{len(all_configs)}")
            
            try:
                result = self.run_single_experiment(config)
                results.append(asdict(result))
                
                # Save intermediate results
                df = pd.DataFrame(results)
                df.to_csv(self.results_csv, index=False)
                print(f"Results saved to {self.results_csv}")
                
                # Also save as JSON for better readability
                results_json = os.path.join(self.output_dir, "profiling_results.json")
                with open(results_json, 'w') as f:
                    json.dump(results, f, indent=2)
                
            except Exception as e:
                print(f"Error in experiment: {e}")
                continue
            
            # Cool down period between experiments
            time.sleep(10)
        
        return results
    
    def analyze_results(self, results_file: str = None):
        """Analyze and visualize profiling results"""
        if results_file is None:
            results_file = self.results_csv
        
        if not os.path.exists(results_file):
            print(f"Results file not found: {results_file}")
            return
            
        df = pd.read_csv(results_file)
        
        # Create summary report
        summary_lines = []
        summary_lines.append("="*60)
        summary_lines.append("PROFILING RESULTS SUMMARY")
        summary_lines.append(f"Generated at: {datetime.now().isoformat()}")
        summary_lines.append(f"Total experiments: {len(df)}")
        summary_lines.append("="*60)
        
        # Best configurations for different metrics
        metrics = ['ttft', 'tbt', 'throughput', 'energy_kwh', 'avg_power_kw']
        
        for metric in metrics:
            if metric in ['ttft', 'tbt', 'energy_kwh', 'avg_power_kw']:
                # Lower is better
                best_idx = df[metric].idxmin()
                summary_lines.append(f"\nBest configuration for {metric} (min):")
            else:
                # Higher is better
                best_idx = df[metric].idxmax()
                summary_lines.append(f"\nBest configuration for {metric} (max):")
            
            best_config = df.iloc[best_idx]
            summary_lines.append(f"  Value: {best_config[metric]:.4f}")
            summary_lines.append(f"  Power cap: {best_config['power_cap']}W")
            summary_lines.append(f"  GPU memory utilization: {best_config['gpu_memory_utilization']}")
            summary_lines.append(f"  Max sequences: {best_config['max_num_seqs']}")
            summary_lines.append(f"  Enable prefix caching: {best_config['enable_prefix_caching']}")
            summary_lines.append(f"  Enable chunked prefill: {best_config['enable_chunked_prefill']}")
            summary_lines.append(f"  Swap space: {best_config['swap_space']} GB")
            summary_lines.append(f"  Max batched tokens: {best_config['max_num_batched_tokens']}")
            summary_lines.append(f"  MPS percentage: {best_config['mps_percentage']}%")
        
        # Statistics for each metric
        summary_lines.append("\n" + "="*60)
        summary_lines.append("METRICS STATISTICS")
        summary_lines.append("="*60)
        
        for metric in metrics:
            summary_lines.append(f"\n{metric}:")
            summary_lines.append(f"  Mean: {df[metric].mean():.4f}")
            summary_lines.append(f"  Std: {df[metric].std():.4f}")
            summary_lines.append(f"  Min: {df[metric].min():.4f}")
            summary_lines.append(f"  Max: {df[metric].max():.4f}")
            summary_lines.append(f"  Median: {df[metric].median():.4f}")
        
        # Correlation analysis
        summary_lines.append("\n" + "="*60)
        summary_lines.append("CORRELATION ANALYSIS")
        summary_lines.append("="*60)
        
        numerical_cols = ['power_cap', 'gpu_memory_utilization', 'max_num_seqs', 
                         'swap_space', 'max_num_batched_tokens', 'mps_percentage',
                         'ttft', 'tbt', 'throughput', 'energy_kwh', 'avg_power_kw']
        
        correlation_matrix = df[numerical_cols].corr()
        
        # Find strong correlations with performance metrics
        for metric in ['ttft', 'tbt', 'throughput', 'energy_kwh']:
            summary_lines.append(f"\nCorrelations with {metric}:")
            correlations = correlation_matrix[metric].sort_values(ascending=False)
            for param, corr in correlations.items():
                if param != metric and abs(corr) > 0.3:
                    summary_lines.append(f"  {param}: {corr:.3f}")
        
        # Top 5 configurations by efficiency (throughput/power)
        df['efficiency'] = df['throughput'] / df['avg_power_kw']
        summary_lines.append("\n" + "="*60)
        summary_lines.append("TOP 5 MOST EFFICIENT CONFIGURATIONS (throughput/power)")
        summary_lines.append("="*60)
        
        top_efficient = df.nlargest(5, 'efficiency')
        for idx, row in top_efficient.iterrows():
            summary_lines.append(f"\nEfficiency: {row['efficiency']:.2f} tokens/s/kW")
            summary_lines.append(f"  Throughput: {row['throughput']:.2f} tokens/s")
            summary_lines.append(f"  Power: {row['avg_power_kw']:.3f} kW")
            summary_lines.append(f"  Config: power_cap={row['power_cap']}W, gpu_mem={row['gpu_memory_utilization']}, ")
            summary_lines.append(f"         max_seqs={row['max_num_seqs']}, mps={row['mps_percentage']}%")
        
        # Save correlation matrix
        corr_file = os.path.join(self.output_dir, "correlation_matrix.csv")
        correlation_matrix.to_csv(corr_file)
        summary_lines.append(f"\nCorrelation matrix saved to: {corr_file}")
        
        # Print to console
        summary_text = '\n'.join(summary_lines)
        print(summary_text)
        
        # Save to file
        with open(self.summary_file, 'w') as f:
            f.write(summary_text)
        print(f"\nSummary report saved to: {self.summary_file}")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='vLLM Inference Profiling Script')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                       help='Model name to profile')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port for vLLM server')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                       help='Number of GPUs for tensor parallelism')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of random configurations to test (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: auto-generated with timestamp)')
    parser.add_argument('--analyze-only', type=str, default=None,
                       help='Path to existing results directory to analyze')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode with single configuration')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Only analyze existing results
        if os.path.isdir(args.analyze_only):
            # It's a directory, look for results file
            results_file = os.path.join(args.analyze_only, "profiling_results.csv")
            summary_file = os.path.join(args.analyze_only, "summary_report.txt")
        else:
            # It's a CSV file
            results_file = args.analyze_only
            summary_file = args.analyze_only.replace('.csv', '_summary.txt')
        
        # Create a temporary profiler just for analysis
        profiler = VLLMProfiler(
            model_name=args.model,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            output_dir=os.path.dirname(results_file) if os.path.dirname(results_file) else "."
        )
        profiler.summary_file = summary_file
        profiler.analyze_results(results_file)
    else:
        profiler = VLLMProfiler(
            model_name=args.model, 
            port=args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            output_dir=args.output_dir
        )
        
        if args.debug:
            # Run single test configuration for debugging
            print("Running in debug mode with minimal configuration...")
            test_config = {
                'power_cap': 250,
                'gpu_memory_utilization': 0.9,
                'max_num_seqs': 128,
                'enable_prefix_caching': False,
                'enable_chunked_prefill': False,
                'swap_space': 4,
                'max_num_batched_tokens': 4096,
                'mps_percentage': 100
            }
            try:
                result = profiler.run_single_experiment(test_config)
                print(f"\nDebug test completed successfully!")
                print(f"TTFT: {result.ttft:.3f}s")
                print(f"TBT: {result.tbt:.3f}s")
                print(f"Throughput: {result.throughput:.2f} tokens/s")
                
                # Save single result
                df = pd.DataFrame([asdict(result)])
                df.to_csv(profiler.results_csv, index=False)
                
                # Analyze the single result
                profiler.analyze_results()
                
            except Exception as e:
                print(f"Debug test failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                profiler.stop_vllm_server()
        else:
            try:
                results = profiler.run_experiments(num_samples=args.num_samples)
                print(f"\nCompleted {len(results)} experiments")
                profiler.analyze_results()
                
                print(f"\n{'='*60}")
                print("ALL RESULTS SAVED TO:")
                print(f"  Directory: {profiler.output_dir}/")
                print(f"  Results CSV: {profiler.results_csv}")
                print(f"  Summary Report: {profiler.summary_file}")
                print(f"  Server Logs: {profiler.output_dir}/logs/")
                print(f"{'='*60}")
                
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                profiler.stop_vllm_server()
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                profiler.stop_vllm_server()
        
        # Cleanup NVML outside of else block
        try:
            pynvml.nvmlShutdown()
        except:
            pass  # Ignore errors during cleanupparse_args()
    
    profiler = VLLMProfiler(
        model_name=args.model, 
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    if args.analyze_only:
        profiler.analyze_results()
    elif args.debug:
        # Run single test configuration for debugging
        print("Running in debug mode with minimal configuration...")
        test_config = {
            'power_cap': 250,
            'gpu_memory_utilization': 0.9,
            'max_num_seqs': 128,
            'enable_prefix_caching': False,
            'enable_chunked_prefill': False,
            'swap_space': 4,
            'max_num_batched_tokens': 4096,
            'mps_percentage': 100
        }
        try:
            result = profiler.run_single_experiment(test_config)
            print(f"\nDebug test completed successfully!")
            print(f"TTFT: {result.ttft:.3f}s")
            print(f"TBT: {result.tbt:.3f}s")
            print(f"Throughput: {result.throughput:.2f} tokens/s")
        except Exception as e:
            print(f"Debug test failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            profiler.stop_vllm_server()
    else:
        try:
            results = profiler.run_experiments(num_samples=args.num_samples)
            print(f"\nCompleted {len(results)} experiments")
            profiler.analyze_results()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            profiler.stop_vllm_server()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            profiler.stop_vllm_server()
        finally:
            # Cleanup
            pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()