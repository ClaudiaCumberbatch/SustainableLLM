"""
scheduler.py - Scheduler implementations
"""

import random
import time
import logging
from typing import List, TYPE_CHECKING
from data_structures import Prompt

if TYPE_CHECKING:
    from server import Server

logger = logging.getLogger(__name__)

class Scheduler:
    """Base scheduler class"""
    
    def __init__(self, servers: List['Server']):
        """
        Initialize scheduler
        
        Args:
            servers: List of available servers
        """
        self.servers = servers
        self.scheduled_prompts = []
        
    def schedule(self, prompt: Prompt) -> 'Server':
        """
        Schedule a prompt to a server (to be overridden by subclasses)
        
        Args:
            prompt: Prompt to schedule
            
        Returns:
            Selected server
        """
        raise NotImplementedError
        
    def get_scheduled_prompts(self):
        """Get all scheduled prompts with their assignments"""
        return self.scheduled_prompts

class RandomScheduler(Scheduler):
    """Random scheduler that randomly selects a server"""
    
    def schedule(self, prompt: Prompt) -> 'Server':
        """
        Randomly select a server to run the prompt
        
        Args:
            prompt: Prompt to schedule
            
        Returns:
            Selected server
        """
        if not self.servers:
            raise ValueError("No servers available for scheduling")
            
        # Randomly select a server
        server = random.choice(self.servers)
        scheduling_time = time.time()
        
        # Record scheduling decision
        self.scheduled_prompts.append({
            'prompt_id': prompt.prompt_id,
            'user_id': prompt.user_id,
            'server_id': server.server_id,
            'scheduling_time': scheduling_time,
            'trace_index': prompt.trace_index
        })
        
        # Calculate scheduling delay
        scheduling_delay = scheduling_time - prompt.arrival_time
        
        # Send prompt to selected server
        server.enqueue_prompt(prompt, scheduling_delay)
        
        logger.info(f"RandomScheduler: Scheduled prompt {prompt.prompt_id} "
                   f"from user {prompt.user_id} to server {server.server_id}")
        
        return server

class RoundRobinScheduler(Scheduler):
    """Round-robin scheduler for even distribution"""
    
    def __init__(self, servers: List['Server']):
        super().__init__(servers)
        self.current_server_index = 0
        
    def schedule(self, prompt: Prompt) -> 'Server':
        """
        Select servers in round-robin fashion
        
        Args:
            prompt: Prompt to schedule
            
        Returns:
            Selected server
        """
        if not self.servers:
            raise ValueError("No servers available for scheduling")
            
        # Select server in round-robin
        server = self.servers[self.current_server_index]
        self.current_server_index = (self.current_server_index + 1) % len(self.servers)
        
        scheduling_time = time.time()
        
        # Record scheduling decision
        self.scheduled_prompts.append({
            'prompt_id': prompt.prompt_id,
            'user_id': prompt.user_id,
            'server_id': server.server_id,
            'scheduling_time': scheduling_time,
            'trace_index': prompt.trace_index
        })
        
        # Calculate scheduling delay
        scheduling_delay = scheduling_time - prompt.arrival_time
        
        # Send prompt to selected server
        server.enqueue_prompt(prompt, scheduling_delay)
        
        logger.info(f"RoundRobinScheduler: Scheduled prompt {prompt.prompt_id} "
                   f"from user {prompt.user_id} to server {server.server_id}")
        
        return server

class LeastLoadedScheduler(Scheduler):
    """Scheduler that selects the server with shortest queue"""
    
    def schedule(self, prompt: Prompt) -> 'Server':
        """
        Select server with the least number of pending prompts
        
        Args:
            prompt: Prompt to schedule
            
        Returns:
            Selected server
        """
        if not self.servers:
            raise ValueError("No servers available for scheduling")
            
        # Find server with minimum queue size
        server = min(self.servers, key=lambda s: s.get_queue_size())
        
        scheduling_time = time.time()
        
        # Record scheduling decision
        self.scheduled_prompts.append({
            'prompt_id': prompt.prompt_id,
            'user_id': prompt.user_id,
            'server_id': server.server_id,
            'scheduling_time': scheduling_time,
            'trace_index': prompt.trace_index
        })
        
        # Calculate scheduling delay
        scheduling_delay = scheduling_time - prompt.arrival_time
        
        # Send prompt to selected server
        server.enqueue_prompt(prompt, scheduling_delay)
        
        logger.info(f"LeastLoadedScheduler: Scheduled prompt {prompt.prompt_id} "
                   f"from user {prompt.user_id} to server {server.server_id} "
                   f"(queue_size={server.get_queue_size()})")
        
        return server