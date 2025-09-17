"""
database_utils.py - Database utilities for creating and managing the profiling database
"""

import sqlite3
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def create_test_database(db_path: str = "profiling_results.db", num_configs: int = 10):
    """
    Create a test database with sample profiling data
    
    Args:
        db_path: Path to the database file
        num_configs: Number of different configurations to generate
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS profiling_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gpu_id INTEGER,
        power_cap INTEGER,
        gpu_memory_utilization REAL,
        max_num_seqs INTEGER,
        enable_prefix_caching BOOLEAN,
        enable_chunked_prefill BOOLEAN,
        swap_space INTEGER,
        max_num_batched_tokens INTEGER,
        mps_percentage INTEGER,
        prompt_id INTEGER,
        prompt TEXT,
        prompt_length INTEGER,
        ttft REAL,
        tbt REAL,
        total_latency REAL,
        tokens_generated INTEGER,
        throughput REAL,
        energy_kwh REAL,
        avg_power_kw REAL,
        peak_power_kw REAL,
        gpu_utilization REAL,
        gpu_memory_used_gb REAL,
        timestamp TEXT,
        generated_text TEXT,
        UNIQUE(gpu_id, prompt_id, power_cap, gpu_memory_utilization, max_num_seqs, 
               enable_prefix_caching, enable_chunked_prefill, swap_space, 
               max_num_batched_tokens, mps_percentage)
    )
    ''')
    
    # Generate sample configurations
    sample_configs = []
    
    # Common configurations that will be used by servers
    base_configs = [
        (300, 0.9, 256, 1, 0, 4, 2048, 100),  # Config 1
        (250, 0.8, 128, 1, 1, 8, 1024, 100),  # Config 2
        (200, 0.7, 64, 0, 1, 16, 512, 100),   # Config 3
    ]
    
    # Generate data for base configurations
    for gpu_id in range(2):  # 2 GPUs
        for config in base_configs:
            power_cap, mem_util, max_seqs, prefix_cache, chunked_prefill, swap, max_tokens, mps = config
            
            # Generate multiple prompt results for each configuration
            for prompt_id in range(5):
                ttft = 0.1 + (prompt_id * 0.01) + (power_cap / 3000)  # Vary by prompt and power
                tbt = 0.02 + (prompt_id * 0.002) + ((300 - power_cap) / 10000)
                tokens = 80 + prompt_id * 10
                
                sample_configs.append((
                    gpu_id, power_cap, mem_util, max_seqs, prefix_cache, chunked_prefill,
                    swap, max_tokens, mps, prompt_id, "", 50 + prompt_id * 10,
                    ttft, tbt, ttft + tokens * tbt, tokens, 1.0/tbt,
                    0.1, 0.25, 0.3, 0.85, 12.5, "2024-01-01", ""
                ))
    
    # Insert all configurations
    for config in sample_configs:
        try:
            cursor.execute('''
            INSERT INTO profiling_results 
            (gpu_id, power_cap, gpu_memory_utilization, max_num_seqs, 
             enable_prefix_caching, enable_chunked_prefill, swap_space, 
             max_num_batched_tokens, mps_percentage, prompt_id, prompt, 
             prompt_length, ttft, tbt, total_latency, tokens_generated, 
             throughput, energy_kwh, avg_power_kw, peak_power_kw, 
             gpu_utilization, gpu_memory_used_gb, timestamp, generated_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', config)
        except sqlite3.IntegrityError:
            pass  # Skip duplicates
            
    conn.commit()
    conn.close()
    logger.info(f"Test database created at {db_path} with {len(sample_configs)} records")
    
def verify_database(db_path: str = "profiling_results.db"):
    """
    Verify database contents and print statistics
    
    Args:
        db_path: Path to the database file
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count total records
        cursor.execute("SELECT COUNT(*) FROM profiling_results")
        total_records = cursor.fetchone()[0]
        
        # Count unique configurations
        cursor.execute("""
            SELECT COUNT(DISTINCT 
                power_cap || '-' || gpu_memory_utilization || '-' || max_num_seqs || '-' ||
                enable_prefix_caching || '-' || enable_chunked_prefill || '-' || swap_space || '-' ||
                max_num_batched_tokens || '-' || mps_percentage
            ) FROM profiling_results
        """)
        unique_configs = cursor.fetchone()[0]
        
        # Get sample data
        cursor.execute("SELECT power_cap, gpu_memory_utilization, ttft, tbt FROM profiling_results LIMIT 5")
        samples = cursor.fetchall()
        
        conn.close()
        
        print(f"\nDatabase Statistics ({db_path}):")
        print(f"  Total records: {total_records}")
        print(f"  Unique configurations: {unique_configs}")
        print(f"\n  Sample records:")
        for sample in samples:
            print(f"    Power: {sample[0]}W, Mem: {sample[1]:.1f}, TTFT: {sample[2]:.3f}s, TBT: {sample[3]:.3f}s")
            
    except sqlite3.Error as e:
        logger.error(f"Database verification error: {e}")
        
def create_trace_file(filename: str = "trace.csv", num_entries: int = 20):
    """
    Create a sample trace CSV file
    
    Args:
        filename: Name of the trace file to create
        num_entries: Number of entries to generate
    """
    import csv
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['creation_time', 'prompt_text'])
        writer.writeheader()
        
        # Generate entries with varying intervals
        current_time = 0.0
        for i in range(num_entries):
            writer.writerow({
                'creation_time': current_time,
                'prompt_text': ''  # Empty as requested
            })
            # Variable intervals between 0.1 and 1.0 seconds
            interval = 0.1 + (i % 10) * 0.09
            current_time += interval
            
    logger.info(f"Created trace file {filename} with {num_entries} entries")