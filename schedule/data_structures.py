"""
data_structures.py - Data structures for the scheduler system
"""

from dataclasses import dataclass

@dataclass
class GPUKnobConfig:
    """GPU configuration knobs"""
    power_cap: int
    gpu_memory_utilization: float
    max_num_seqs: int
    enable_prefix_caching: bool
    enable_chunked_prefill: bool
    swap_space: int
    max_num_batched_tokens: int
    mps_percentage: int

@dataclass
class Prompt:
    """Prompt data structure"""
    prompt_id: int
    prompt_text: str  # Will be empty string
    creation_time: float
    arrival_time: float  # Time when prompt arrives at scheduler
    user_id: int
    trace_index: int  # Index in the original trace file

@dataclass
class PerformanceMetrics:
    """Performance metrics for a completed request"""
    prompt_id: int
    server_id: int
    ttft: float  # Time to first token
    ttlt: float  # Time to last token
    tbt: float   # Time between tokens
    throughput: float
    tokens_generated: int
    scheduling_delay: float
    rtt: float
    user_id: int
    trace_index: int