"""
main.py - Main execution script for the scheduler simulation
"""

import time
import logging
import argparse
from typing import List

from data_structures import GPUKnobConfig
from trace_manager import TraceManager
from user import User
from scheduler import RandomScheduler, RoundRobinScheduler, LeastLoadedScheduler
from server import Server
from database_utils import create_test_database, verify_database, create_trace_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_servers(num_servers: int = 2) -> List[Server]:
    """
    Initialize multiple servers with different GPU configurations
    
    Args:
        num_servers: Number of servers to create
        
    Returns:
        List of Server instances
    """
    servers = []
    
    # Predefined configurations
    configs = [
        GPUKnobConfig(
            power_cap=300,
            gpu_memory_utilization=0.9,
            max_num_seqs=256,
            enable_prefix_caching=True,
            enable_chunked_prefill=False,
            swap_space=4,
            max_num_batched_tokens=4096,
            mps_percentage=100
        ),
        GPUKnobConfig(
            power_cap=250,
            gpu_memory_utilization=0.85,
            max_num_seqs=128,
            enable_prefix_caching=True,
            enable_chunked_prefill=False,
            swap_space=8,
            max_num_batched_tokens=4096,
            mps_percentage=100
        ),
        GPUKnobConfig(
            power_cap=200,
            gpu_memory_utilization=0.7,
            max_num_seqs=64,
            enable_prefix_caching=True,
            enable_chunked_prefill=False,
            swap_space=8,
            max_num_batched_tokens=4096,
            mps_percentage=100
        ),
    ]
    
    for i in range(num_servers):
        # Use configuration cyclically if more servers than configs
        config = configs[i % len(configs)]
        # Vary RTT slightly for different server locations
        location_rtt = 0.01 + (i * 0.005)
        
        server = Server(
            server_id=i,
            gpu_knob_config=config,
            location_rtt=location_rtt
        )
        servers.append(server)
        
        logger.info(f"Initialized Server {i} with power_cap={config.power_cap}W, "
                   f"RTT={location_rtt:.3f}s")
    
    return servers

def run_simulation(args):
    """
    Run the main simulation
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*80)
    print("CPU NODE SCHEDULER SIMULATION")
    print("="*80)
    
    # Create test database if needed
    if args.create_db:
        create_test_database()
        verify_database()
    
    # Create trace file if needed
    if args.create_trace:
        create_trace_file(args.trace_file, args.trace_entries)
    
    # Initialize trace manager with optional limit
    print(f"\nInitializing trace manager with {'first ' + str(args.max_entries) + ' entries' if args.max_entries else 'all entries'}...")
    trace_manager = TraceManager(args.trace_file, max_entries=args.max_entries)
    total_entries = trace_manager.get_total_entries()
    print(f"Trace manager loaded {total_entries} entries")
    
    # Initialize servers
    print(f"\nInitializing {args.num_servers} servers...")
    servers = initialize_servers(args.num_servers)
    
    # Initialize scheduler based on policy
    print(f"\nInitializing {args.scheduler} scheduler...")
    if args.scheduler == "random":
        scheduler = RandomScheduler(servers)
    elif args.scheduler == "roundrobin":
        scheduler = RoundRobinScheduler(servers)
    elif args.scheduler == "leastloaded":
        scheduler = LeastLoadedScheduler(servers)
    else:
        raise ValueError(f"Unknown scheduler type: {args.scheduler}")
    
    # Initialize users
    print(f"\nInitializing {args.num_users} users...")
    users = []
    user_threads = []
    
    for i in range(args.num_users):
        user = User(
            user_id=i,
            trace_manager=trace_manager,
            scheduler=scheduler,
            rtt=0.01
        )
        users.append(user)
    
    # Start all users sending prompts
    print("\nStarting simulation...")
    start_time = time.time()
    
    for user in users:
        thread = user.start_sending()
        user_threads.append(thread)
    
    # Wait for all users to finish sending
    for i, thread in enumerate(user_threads):
        thread.join()
        print(f"User {i} completed sending")
    
    # Wait for processing to complete
    print("\nWaiting for servers to complete processing...")
    max_wait = 10  # Maximum wait time in seconds
    wait_start = time.time()
    
    while time.time() - wait_start < max_wait:
        # Check if all servers have empty queues
        all_empty = all(server.get_queue_size() == 0 for server in servers)
        if all_empty:
            time.sleep(1)  # Extra time for final processing
            break
        time.sleep(0.5)
    
    simulation_time = time.time() - start_time
    
    # Collect and display results
    print_results(users, servers, scheduler, simulation_time)
    
    # Stop all servers
    for server in servers:
        server.stop()
    
    print("\nSimulation completed!")

def print_results(users: List[User], servers: List[Server], scheduler, simulation_time: float):
    """
    Print simulation results and statistics
    
    Args:
        users: List of User instances
        servers: List of Server instances
        scheduler: Scheduler instance
        simulation_time: Total simulation time
    """
    print("\n" + "="*80)
    print("SIMULATION RESULTS")
    print("="*80)
    
    # User statistics
    print("\nUser Statistics:")
    total_sent = 0
    for user in users:
        sent_count = len(user.get_sent_prompts())
        total_sent += sent_count
        print(f"  User {user.user_id}: Sent {sent_count} prompts")
    print(f"  Total prompts sent: {total_sent}")
    
    # Server statistics
    print("\nServer Statistics:")
    all_metrics = []
    for server in servers:
        metrics = server.get_metrics()
        all_metrics.extend(metrics)
        
        if metrics:
            avg_ttft = sum(m.ttft for m in metrics) / len(metrics)
            avg_ttlt = sum(m.ttlt for m in metrics) / len(metrics)
            avg_throughput = sum(m.throughput for m in metrics) / len(metrics)
            avg_scheduling_delay = sum(m.scheduling_delay for m in metrics) / len(metrics)
            
            print(f"\n  Server {server.server_id}:")
            print(f"    Processed: {len(metrics)} requests")
            print(f"    Average TTFT: {avg_ttft:.3f}s")
            print(f"    Average TTLT: {avg_ttlt:.3f}s")
            print(f"    Average Throughput: {avg_throughput:.2f} tokens/s")
            print(f"    Average Scheduling Delay: {avg_scheduling_delay:.3f}s")
        else:
            print(f"\n  Server {server.server_id}: No requests processed")
    
    # Overall statistics
    if all_metrics:
        print("\n" + "-"*40)
        print("Overall Performance Metrics:")
        print("-"*40)
        
        overall_avg_ttft = sum(m.ttft for m in all_metrics) / len(all_metrics)
        overall_avg_ttlt = sum(m.ttlt for m in all_metrics) / len(all_metrics)
        overall_avg_throughput = sum(m.throughput for m in all_metrics) / len(all_metrics)
        overall_avg_scheduling = sum(m.scheduling_delay for m in all_metrics) / len(all_metrics)
        
        print(f"  Total requests processed: {len(all_metrics)}")
        print(f"  Average TTFT: {overall_avg_ttft:.3f}s")
        print(f"  Average TTLT: {overall_avg_ttlt:.3f}s")
        print(f"  Average Throughput: {overall_avg_throughput:.2f} tokens/s")
        print(f"  Average Scheduling Delay: {overall_avg_scheduling:.3f}s")
        print(f"  Total simulation time: {simulation_time:.2f}s")
        
        # Performance breakdown
        print("\nPerformance Breakdown:")
        avg_rtt = sum(m.rtt for m in all_metrics) / len(all_metrics)
        avg_tbt = sum(m.tbt for m in all_metrics) / len(all_metrics)
        print(f"  Average RTT component: {avg_rtt:.3f}s")
        print(f"  Average TBT: {avg_tbt:.3f}s")
        
        # Distribution analysis
        print("\nRequest Distribution:")
        for server in servers:
            server_metrics = [m for m in all_metrics if m.server_id == server.server_id]
            percentage = (len(server_metrics) / len(all_metrics)) * 100 if all_metrics else 0
            print(f"  Server {server.server_id}: {len(server_metrics)} requests ({percentage:.1f}%)")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='CPU Node Scheduler Simulation')
    
    # Simulation parameters
    parser.add_argument('--num-users', type=int, default=3,
                       help='Number of users (default: 3)')
    parser.add_argument('--num-servers', type=int, default=2,
                       help='Number of servers (default: 2)')
    parser.add_argument('--scheduler', choices=['random', 'roundrobin', 'leastloaded'],
                       default='random', help='Scheduler policy (default: random)')
    
    # Trace file parameters
    parser.add_argument('--trace-file', type=str, default='../dataset/alibaba_2025/filtered_hn_data.csv',
                       help='Path to trace file (default: ../dataset/alibaba_2025/filtered_hn_data.csv)')
    parser.add_argument('--max-entries', type=int, default=5,
                       help='Maximum number of trace entries to use (default: all)')
    
    # Database and trace creation
    parser.add_argument('--create-db', action='store_true',
                       help='Create test database')
    parser.add_argument('--create-trace', action='store_true',
                       help='Create sample trace file')
    parser.add_argument('--trace-entries', type=int, default=20,
                       help='Number of entries in sample trace file (default: 20)')
    
    # Logging level
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run simulation
    try:
        run_simulation(args)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation error: {e}", exc_info=True)

if __name__ == "__main__":
    main()