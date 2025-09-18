"""
trace_manager.py - Manages trace file reading and distribution to users
"""

import csv
import threading
import logging
from typing import List, Optional
from data_structures import Prompt

logger = logging.getLogger(__name__)

class TraceManager:
    """Manages a single trace file and distributes entries to multiple users"""
    
    def __init__(self, trace_file: str, max_entries: Optional[int] = None):
        """
        Initialize trace manager
        
        Args:
            trace_file: Path to the trace CSV file
            max_entries: Maximum number of entries to use (for testing). None means use all.
        """
        self.trace_file = trace_file
        self.max_entries = max_entries
        self.trace_data = []
        self.current_index = 0
        self.lock = threading.Lock()
        self._load_trace()
        
    def _load_trace(self):
        """Load trace data from CSV file"""
        try:
            with open(self.trace_file, 'r') as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    # Check if we've reached max_entries
                    if self.max_entries and idx >= self.max_entries:
                        break
                        
                    creation_time = float(row.get('creation_time', 0))
                    self.trace_data.append({
                        'creation_time': creation_time,
                        'trace_index': idx
                    })
                    
            logger.info(f"Loaded {len(self.trace_data)} entries from {self.trace_file}")
            
        except FileNotFoundError:
            logger.warning(f"Trace file {self.trace_file} not found. Creating dummy data.")
            # Create dummy trace data for testing
            num_entries = self.max_entries if self.max_entries else 30
            for i in range(num_entries):
                self.trace_data.append({
                    'creation_time': i * 0.5,  # 0.5 second intervals
                    'trace_index': i
                })
            logger.info(f"Created {len(self.trace_data)} dummy entries")
            
    def get_next_entry(self, user_id: int) -> Optional[Prompt]:
        """
        Get the next trace entry for a user (round-robin distribution)
        
        Args:
            user_id: ID of the requesting user
            
        Returns:
            Prompt object or None if no more entries
        """
        with self.lock:
            if self.current_index >= len(self.trace_data):
                return None
                
            entry = self.trace_data[self.current_index]
            prompt = Prompt(
                prompt_id=self.current_index,
                prompt_text="",  # Empty string as requested
                creation_time=entry['creation_time']*0.001, # TODO: this is only for test
                arrival_time=0,  # Will be set when sent to scheduler
                user_id=user_id,
                trace_index=entry['trace_index']
            )
            
            self.current_index += 1
            return prompt
            
    def reset(self):
        """Reset the trace manager to start from beginning"""
        with self.lock:
            self.current_index = 0
            
    def get_total_entries(self) -> int:
        """Get total number of trace entries"""
        return len(self.trace_data)
        
    def get_remaining_entries(self) -> int:
        """Get number of remaining entries"""
        with self.lock:
            return len(self.trace_data) - self.current_index