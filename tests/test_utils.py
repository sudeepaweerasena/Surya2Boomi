#!/usr/bin/env python3
"""
Performance Testing Utilities

Shared utilities for timing, statistics, logging, and result management.
"""

import time
import json
import statistics
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import List, Dict, Any, Callable
import psutil
import os


class Timer:
    """High-precision timer for measuring execution time"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None
        
    def start(self):
        """Start the timer"""
        self.start_time = time.perf_counter()
        return self
        
    def stop(self):
        """Stop the timer and calculate elapsed time"""
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, *args):
        self.stop()


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        timer = Timer()
        timer.start()
        result = func(*args, **kwargs)
        elapsed = timer.stop()
        return result, elapsed
    return wrapper


@contextmanager
def measure_time(description: str = ""):
    """Context manager to measure and print execution time"""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if description:
        print(f"⏱️  {description}: {elapsed:.4f}s")
    else:
        print(f"⏱️  Elapsed time: {elapsed:.4f}s")


class PerformanceMetrics:
    """Calculate statistical metrics from timing data"""
    
    @staticmethod
    def calculate(times: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics from a list of timing measurements"""
        if not times:
            return {}
            
        sorted_times = sorted(times)
        n = len(times)
        
        return {
            'count': n,
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'stdev': statistics.stdev(times) if n > 1 else 0,
            'p50': sorted_times[int(n * 0.50)],
            'p75': sorted_times[int(n * 0.75)],
            'p90': sorted_times[int(n * 0.90)],
            'p95': sorted_times[int(n * 0.95)],
            'p99': sorted_times[int(n * 0.99)] if n >= 100 else sorted_times[-1],
        }
    
    @staticmethod
    def format_stats(stats: Dict[str, float], unit: str = "s") -> str:
        """Format statistics for display"""
        lines = [
            f"Count:  {stats.get('count', 0)}",
            f"Mean:   {stats.get('mean', 0):.4f}{unit}",
            f"Median: {stats.get('median', 0):.4f}{unit}",
            f"Min:    {stats.get('min', 0):.4f}{unit}",
            f"Max:    {stats.get('max', 0):.4f}{unit}",
            f"StdDev: {stats.get('stdev', 0):.4f}{unit}",
            f"P50:    {stats.get('p50', 0):.4f}{unit}",
            f"P95:    {stats.get('p95', 0):.4f}{unit}",
            f"P99:    {stats.get('p99', 0):.4f}{unit}",
        ]
        return "\n".join(lines)


class ResourceMonitor:
    """Monitor system resource usage during tests"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = None
        self.peak_memory = 0
        self.cpu_samples = []
        
    def start(self):
        """Start monitoring resources"""
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        self.cpu_samples = []
        return self
        
    def sample(self):
        """Take a sample of current resource usage"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        
        cpu_percent = self.process.cpu_percent(interval=0.1)
        self.cpu_samples.append(cpu_percent)
        
    def get_stats(self) -> Dict[str, float]:
        """Get resource usage statistics"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'initial_memory_mb': self.initial_memory,
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': current_memory - self.initial_memory,
            'avg_cpu_percent': statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            'peak_cpu_percent': max(self.cpu_samples) if self.cpu_samples else 0,
        }


class TestResults:
    """Manage and save test results"""
    
    def __init__(self, test_name: str, output_dir: str = "tests/results"):
        self.test_name = test_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now(timezone.utc)
        self.results = {
            'test_name': test_name,
            'timestamp': self.timestamp.isoformat(),
            'metrics': {},
            'metadata': {}
        }
        
    def add_metric(self, key: str, value: Any):
        """Add a metric to the results"""
        self.results['metrics'][key] = value
        
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the results"""
        self.results['metadata'][key] = value
        
    def save_json(self):
        """Save results as JSON"""
        filename = f"{self.test_name}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"📁 Results saved to: {filepath}")
        return filepath
        
    def save_summary(self):
        """Save a human-readable summary"""
        filename = f"{self.test_name}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}_summary.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(f"Performance Test Results: {self.test_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
            
            f.write("METRICS:\n")
            f.write("-" * 80 + "\n")
            for key, value in self.results['metrics'].items():
                if isinstance(value, dict):
                    f.write(f"\n{key}:\n")
                    for k, v in value.items():
                        if isinstance(v, float):
                            f.write(f"  {k}: {v:.4f}\n")
                        else:
                            f.write(f"  {k}: {v}\n")
                elif isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            if self.results['metadata']:
                f.write("\n\nMETADATA:\n")
                f.write("-" * 80 + "\n")
                for key, value in self.results['metadata'].items():
                    f.write(f"{key}: {value}\n")
        
        print(f"📄 Summary saved to: {filepath}")
        return filepath


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{title}")
    print("-" * 80)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


def format_throughput(count: int, duration: float) -> str:
    """Format throughput as items/second"""
    if duration > 0:
        rate = count / duration
        return f"{rate:.2f} items/sec"
    return "N/A"
