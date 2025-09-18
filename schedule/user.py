"""
user.py - User class implementation
"""

import time
import threading
import logging
from typing import TYPE_CHECKING, List
from trace_manager import TraceManager
from data_structures import Prompt

if TYPE_CHECKING:
    from scheduler import Scheduler

logger = logging.getLogger(__name__)

class User:
    """User class that gets prompts from shared trace and sends them to scheduler"""
    
    def __init__(self, user_id: int, trace_manager: TraceManager, 
                 scheduler: 'Scheduler', rtt: float = 0.01):
        """
        Initialize user
        
        Args:
            user_id: Unique user identifier
            trace_manager: Shared trace manager instance
            scheduler: Scheduler instance
            rtt: Round trip time in seconds
        """
        self.user_id = user_id
        self.trace_manager = trace_manager
        self.scheduler = scheduler
        self.rtt = rtt
        self.prompts_sent = []
        self.thread = None
        
    def start_sending(self) -> threading.Thread:
        """Start sending prompts to scheduler"""
        self.thread = threading.Thread(target=self._send_prompts)
        self.thread.start()
        return self.thread
        
    def _send_prompts(self):
        """Send prompts to scheduler respecting creation time intervals"""
        last_creation_time = 0
        start_time = time.time()
        
        while True:
            # Get next prompt from trace manager
            prompt = self.trace_manager.get_next_entry(self.user_id)
            
            if prompt is None:
                logger.info(f"User {self.user_id} finished sending prompts")
                break
                
            # Calculate and sleep for the interval
            interval = prompt.creation_time - last_creation_time
            if interval > 0:
                time.sleep(interval)
                
            # Set arrival time and send to scheduler
            prompt.arrival_time = time.time()
            self.scheduler.schedule(prompt)
            
            # Record sent prompt
            self.prompts_sent.append(prompt)
            
            elapsed = time.time() - start_time
            logger.info(f"User {self.user_id} sent prompt {prompt.prompt_id} "
                       f"(trace_index={prompt.trace_index}) at {elapsed:.3f}s")
            
            last_creation_time = prompt.creation_time
            
    def wait_completion(self):
        """Wait for the user thread to complete"""
        if self.thread:
            self.thread.join()
            
    def get_sent_prompts(self) -> List[Prompt]:
        """Get list of prompts sent by this user"""
        return self.prompts_sent