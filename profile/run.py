"""
Main runner script for vLLM profiling experiments
This script imports and uses classes from vllm_profiling.py
"""

import os
import sys
import argparse
import numpy as np
import pynvml
from datetime import datetime
import traceback

# Import all classes from vllm_profiling module
from new import (
    VLLMProfiler,
    DatasetLoader,
    DatabaseManager,
    PowerMonitor,
    ProfilingResult
)

def validate_files(trace_file: str, prompt_file: str) -> bool:
    """Validate that required input files exist"""
    if not os.path.exists(trace_file):
        print(f"❌ Error: Trace file not found: {trace_file}")
        return False
    
    if not os.path.exists(prompt_file):
        print(f"❌ Error: Prompt file not found: {prompt_file}")
        return False
    
    print(f"✅ Found trace file: {trace_file}")
    print(f"✅ Found prompt file: {prompt_file}")
    return True

def run_single_config_test(args):
    """Run a single configuration test for debugging"""
    print("\n" + "="*70)
    print("RUNNING SINGLE CONFIGURATION TEST")
    print("="*70)
    
    # Initialize profiler
    profiler = VLLMProfiler(
        model_name=args.model,
        port=args.port,
        trace_file=args.trace_file,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir
    )
    
    # Define test configuration
    test_config = {
        'power_cap': args.power_cap if hasattr(args, 'power_cap') else 200,
        'gpu_memory_utilization': args.gpu_mem if hasattr(args, 'gpu_mem') else 0.9,
        'max_num_seqs': args.max_seqs if hasattr(args, 'max_seqs') else 128,
        'enable_prefix_caching': True,
        'enable_chunked_prefill': False,
        'swap_space': 4,
        'max_num_batched_tokens': 4096,
        'mps_percentage': 100
    }
    
    print("\nTest Configuration:")
    for key, value in test_config.items():
        print(f"  {key}: {value}")
    
    # Display timing settings
    time_scale = args.time_scale if hasattr(args, 'time_scale') else 1.0
    print(f"\nTiming Settings:")
    print(f"  Time scale: {time_scale}")
    if time_scale == 1.0:
        print(f"  Mode: Real-time replay (matching trace timing)")
    elif time_scale < 1.0:
        print(f"  Mode: Accelerated ({1/time_scale:.1f}x faster than trace)")
    else:
        print(f"  Mode: Slowed ({time_scale:.1f}x slower than trace)")
    
    try:
        # Run experiment with limited trace records
        trace_limit = args.trace_limit or 5
        print(f"\nProcessing {trace_limit} trace records...")
        
        results = profiler.run_config_experiment(
            test_config, 
            trace_limit=trace_limit,
            time_scale=time_scale
        )
        
        # Print summary statistics
        print(f"\n{'='*70}")
        print("TEST RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Requests processed: {len(results)}")
        
        if results:
            avg_ttft = np.mean([r.ttft for r in results])
            avg_tbt = np.mean([r.tbt for r in results])
            avg_throughput = np.mean([r.throughput for r in results])
            avg_energy = np.mean([r.energy_kwh for r in results])
            
            print(f"Average TTFT: {avg_ttft:.4f} seconds")
            print(f"Average TBT: {avg_tbt:.4f} seconds")
            print(f"Average Throughput: {avg_throughput:.2f} tokens/second")
            print(f"Average Energy: {avg_energy:.6f} kWh")
        
        # Export results
        profiler.export_results_to_csv()
        profiler.generate_analysis_report()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False
    finally:
        profiler.stop_vllm_server()
        profiler.db_manager.close()

def run_full_profiling(args):
    """Run full profiling experiments across all configurations"""
    print("\n" + "="*70)
    print("RUNNING FULL PROFILING EXPERIMENTS")
    print("="*70)
    
    # Initialize profiler
    profiler = VLLMProfiler(
        model_name=args.model,
        port=args.port,
        trace_file=args.trace_file,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir
    )
    
    # Override parameter ranges if custom values provided
    if args.custom_params:
        print("\nUsing custom parameter ranges...")
        # Parse custom parameters (format: key=value1,value2,...)
        for param in args.custom_params:
            key, values = param.split('=')
            values = [eval(v) for v in values.split(',')]
            profiler.param_ranges[key] = values
            print(f"  {key}: {values}")
    
    print(f"\nTotal configurations to test: {calculate_total_configs(profiler.param_ranges)}")
    
    # Display timing settings
    time_scale = args.time_scale if hasattr(args, 'time_scale') else 1.0
    print(f"\nTiming Settings:")
    print(f"  Time scale: {time_scale}")
    if time_scale == 1.0:
        print(f"  Mode: Real-time replay (matching trace timing)")
    elif time_scale < 1.0:
        print(f"  Mode: Accelerated ({1/time_scale:.1f}x faster than trace)")
    else:
        print(f"  Mode: Slowed ({time_scale:.1f}x slower than trace)")
    
    if args.trace_limit:
        print(f"  Limiting to {args.trace_limit} trace records per configuration")
    
    try:
        # Run all experiments
        results = profiler.run_all_experiments(
            trace_limit=args.trace_limit,
            time_scale=time_scale
        )
        
        print(f"\n{'='*70}")
        print("PROFILING COMPLETE")
        print(f"{'='*70}")
        print(f"Total requests processed: {len(results)}")
        
        # Generate comprehensive analysis
        profiler.generate_analysis_report()
        profiler.export_results_to_csv()
        
        # Print location of results
        print_results_location(profiler.output_dir)
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error during profiling: {e}")
        traceback.print_exc()
        return False
    finally:
        profiler.stop_vllm_server()
        profiler.db_manager.close()
        cleanup_resources()

def analyze_existing_results(args):
    """Analyze existing profiling results from database"""
    print("\n" + "="*70)
    print("ANALYZING EXISTING RESULTS")
    print("="*70)
    
    # Determine database path
    if os.path.isdir(args.analyze_only):
        db_path = os.path.join(args.analyze_only, "profiling_results.db")
        output_dir = args.analyze_only
    else:
        db_path = args.analyze_only
        output_dir = os.path.dirname(db_path) or "."
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False
    
    print(f"✅ Found database: {db_path}")
    
    try:
        # Create profiler instance for analysis
        # Use dummy file paths since we're only analyzing
        profiler = VLLMProfiler(
            model_name=args.model,
            port=args.port,
            trace_file=args.trace_file or "trace.csv",
            prompt_file=args.prompt_file or "train.jsonl",
            output_dir=output_dir
        )
        
        # Generate analysis report
        profiler.generate_analysis_report()
        profiler.export_results_to_csv()
        
        print_results_location(output_dir)
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        traceback.print_exc()
        return False

def calculate_total_configs(param_ranges):
    """Calculate total number of configurations"""
    total = 1
    for key, values in param_ranges.items():
        total *= len(values)
    return total

def print_results_location(output_dir):
    """Print the location of all result files"""
    print(f"\n{'='*70}")
    print("📁 RESULTS LOCATION:")
    print(f"{'='*70}")
    print(f"  Directory: {output_dir}/")
    print(f"  Database: {os.path.join(output_dir, 'profiling_results.db')}")
    print(f"  Detailed CSV: {os.path.join(output_dir, 'detailed_results.csv')}")
    print(f"  Summary CSV: {os.path.join(output_dir, 'config_summary.csv')}")
    print(f"  Analysis Report: {os.path.join(output_dir, 'analysis_report.txt')}")
    print(f"  Logs: {os.path.join(output_dir, 'logs/')}")
    print(f"{'='*70}")

def cleanup_resources():
    """Clean up system resources"""
    try:
        pynvml.nvmlShutdown()
    except:
        pass

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='vLLM Profiling with Real Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single configuration test with 5 requests
  %(prog)s --single-config --trace-limit 5
  
  # Run full profiling with custom trace/prompt files
  %(prog)s --trace-file mydata/trace.csv --prompt-file mydata/prompts.jsonl
  
  # Analyze existing results
  %(prog)s --analyze-only ./results_20241201/
  
  # Run with custom parameters
  %(prog)s --custom-params "power_cap=100,200,300" "max_num_seqs=64,128"
        """
    )
    
    # Basic arguments
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                       help='Model name to profile (default: meta-llama/Llama-2-7b-hf)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port for vLLM server (default: 8000)')
    
    # Input files
    parser.add_argument('--trace-file', type=str, default='../dataset/alibaba_2025/filtered_data.csv',
                   help='Path to trace CSV file')
    parser.add_argument('--prompt-file', type=str, default='../dataset/grad_school_math/train.jsonl',
                   help='Path to prompt JSONL file')

    
    # Execution modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--single-config', action='store_true',
                           help='Run single configuration for testing')
    mode_group.add_argument('--analyze-only', type=str, metavar='PATH',
                           help='Analyze existing database/directory')
    
    # Experiment parameters
    parser.add_argument('--trace-limit', type=int, default=None,
                       help='Limit number of trace records per config')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: auto-generated)')
    
    # Timing control
    parser.add_argument('--time-scale', type=float, default=1,
                       help='Time scale for request timing (1.0=real-time, 0.1=10x faster, 0.01=100x faster)')

    # Custom configuration parameters
    parser.add_argument('--custom-params', nargs='+', metavar='KEY=VAL1,VAL2',
                       help='Custom parameter ranges (e.g., power_cap=100,200,300)')
    
    # Single config test parameters
    test_group = parser.add_argument_group('single-config test parameters')
    test_group.add_argument('--power-cap', type=int, default=200,
                          help='Power cap for single-config test (default: 200W)')
    test_group.add_argument('--gpu-mem', type=float, default=0.9,
                          help='GPU memory utilization for test (default: 0.9)')
    test_group.add_argument('--max-seqs', type=int, default=128,
                          help='Max sequences for test (default: 128)')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()

    # Validate time scale
    if args.time_scale <= 0:
        print("❌ Error: Time scale must be positive")
        sys.exit(1)
    
    # Print header
    if not args.quiet:
        print("\n" + "="*70)
        print(" vLLM PROFILING SYSTEM ".center(70))
        print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(70))
        print("="*70)

        # Display timing mode
        if args.time_scale == 1.0:
            print(" Timing Mode: REAL-TIME (matching trace timing) ".center(70))
        elif args.time_scale < 1.0:
            speed_factor = 1.0 / args.time_scale
            print(f" Timing Mode: ACCELERATED ({speed_factor:.1f}x faster) ".center(70))
        else:
            print(f" Timing Mode: SLOWED ({args.time_scale:.1f}x slower) ".center(70))
        print("="*70)
    
    # Execute based on mode
    success = False
    
    if args.analyze_only:
        # Analyze existing results
        success = analyze_existing_results(args)
        
    else:
        # Validate input files
        if not validate_files(args.trace_file, args.prompt_file):
            sys.exit(1)
        
        if args.single_config:
            # Run single configuration test
            success = run_single_config_test(args)
        else:
            # Run full profiling
            success = run_full_profiling(args)
    
    # Print footer
    if not args.quiet:
        print("\n" + "="*70)
        if success:
            print(" ✅ EXECUTION COMPLETED SUCCESSFULLY ".center(70))
        else:
            print(" ❌ EXECUTION FAILED ".center(70))
        print(f" Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(70))
        print("="*70 + "\n")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()