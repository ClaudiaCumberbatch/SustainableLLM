"""
server.py - Server class implementation
"""

import sqlite3
import random
import threading
import logging
from queue import Queue
from typing import List, Optional, Dict
from data_structures import GPUKnobConfig, Prompt, PerformanceMetrics

logger = logging.getLogger(__name__)

class Server:
    """Server class with GPU knob configuration and prompt processing"""
    
    def __init__(self, server_id: int, gpu_knob_config: GPUKnobConfig, 
                 db_path: str = "/home/cc/profile_dataset/V100_alpaca/profiling_results.db", location_rtt: float = 0.01):
        """
        Initialize server
        
        Args:
            server_id: Unique server identifier
            gpu_knob_config: GPU configuration settings
            db_path: Path to the profiling database
            location_rtt: RTT based on server location
        """
        self.server_id = server_id
        self.gpu_knob_config = gpu_knob_config
        self.db_path = db_path
        self.location_rtt = location_rtt
        self.prompt_queue = Queue()
        self.completed_requests = []
        self.running = True
        self.processing_thread = None
        self._init_db()
        self._start_processing()
        
    def _init_db(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
    def enqueue_prompt(self, prompt: Prompt, scheduling_delay: float):
        """
        Add a prompt to the processing queue
        
        Args:
            prompt: Prompt to process
            scheduling_delay: Delay introduced by scheduling
        """
        self.prompt_queue.put((prompt, scheduling_delay))
        
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.prompt_queue.qsize()
        
    def _start_processing(self):
        """Start the prompt processing thread"""
        self.processing_thread = threading.Thread(target=self._process_prompts)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def _process_prompts(self):
        """Process prompts from the queue"""
        while self.running:
            try:
                prompt, scheduling_delay = self.prompt_queue.get(timeout=1)
                metrics = self._process_single_prompt(prompt, scheduling_delay)
                self.completed_requests.append(metrics)
                logger.debug(f"Server {self.server_id} completed prompt {prompt.prompt_id}")
            except:
                continue
                
    def _process_single_prompt(self, prompt: Prompt, scheduling_delay: float) -> PerformanceMetrics:
        """
        Process a single prompt and calculate performance metrics
        
        Args:
            prompt: Prompt to process
            scheduling_delay: Scheduling delay
            
        Returns:
            Performance metrics for the processed prompt
        """
        # Query profile database for matching configuration
        profile_data = self._query_profile_data(prompt)
        
        if profile_data:
            # Extract metrics from profile data
            ttft_profile = profile_data['ttft']
            tbt = profile_data['tbt']
            tokens_generated = profile_data['tokens_generated']
            
            # Calculate final performance metrics
            # TTFT = RTT + scheduling_delay + profile_TTFT
            ttft = self.location_rtt + scheduling_delay + ttft_profile
            
            # TTLT = TTFT + tokens * tbt
            ttlt = ttft + (tokens_generated * tbt)
            
            # Throughput = 1/tbt
            throughput = 1.0 / tbt if tbt > 0 else 0
            
        else:
            # Use default values if no profile data found
            logger.warning(f"No profile data found for prompt {prompt.prompt_id} on server {self.server_id}")
            ttft = self.location_rtt + scheduling_delay + 0.1  # Default 100ms
            tbt = 0.05  # Default 50ms per token
            tokens_generated = 100  # Default tokens
            ttlt = ttft + (tokens_generated * tbt)
            throughput = 1.0 / tbt
            
        metrics = PerformanceMetrics(
            prompt_id=prompt.prompt_id,
            server_id=self.server_id,
            ttft=ttft,
            ttlt=ttlt,
            tbt=tbt,
            throughput=throughput,
            tokens_generated=tokens_generated,
            scheduling_delay=scheduling_delay,
            rtt=self.location_rtt,
            user_id=prompt.user_id,
            trace_index=prompt.trace_index
        )
        
        logger.info(f"Server {self.server_id} processed prompt {prompt.prompt_id}: "
                   f"TTFT={ttft:.3f}s, TTLT={ttlt:.3f}s, Throughput={throughput:.2f} tokens/s")
        
        return metrics
        
    def _query_profile_data(self, prompt: Prompt) -> Optional[Dict]:
        """
        Query profile database for matching knob configuration
        
        Args:
            prompt: Prompt being processed
            
        Returns:
            Dictionary with profile data or None if not found
        """
        query = """
        SELECT ttft, tbt, total_latency, tokens_generated, throughput
        FROM profiling_results
        WHERE power_cap = ?
          AND gpu_memory_utilization = ?
          AND max_num_seqs = ?
          AND enable_prefix_caching = ?
          AND enable_chunked_prefill = ?
          AND swap_space = ?
          AND max_num_batched_tokens = ?
          AND mps_percentage = ?
        """
        
        params = (
            self.gpu_knob_config.power_cap,
            self.gpu_knob_config.gpu_memory_utilization,
            self.gpu_knob_config.max_num_seqs,
            self.gpu_knob_config.enable_prefix_caching,
            self.gpu_knob_config.enable_chunked_prefill,
            self.gpu_knob_config.swap_space,
            self.gpu_knob_config.max_num_batched_tokens,
            self.gpu_knob_config.mps_percentage
        )
        
        try:
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            
            if results:
                # Randomly select one if multiple records found
                selected = random.choice(results)
                return {
                    'ttft': selected[0],
                    'tbt': selected[1],
                    'total_latency': selected[2],
                    'tokens_generated': selected[3],
                    'throughput': selected[4]
                }
        except sqlite3.Error as e:
            logger.error(f"Database query error: {e}")
            
        return None
        
    def stop(self):
        """Stop the server processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        self.conn.close()
        
    def get_metrics(self) -> List[PerformanceMetrics]:
        """Get all completed request metrics"""
        return self.completed_requests