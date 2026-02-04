#!/usr/bin/env python3
"""
Comprehensive Performance Testing Suite

Tests the following metrics:
1. Response Time - Latency of forecast generation and predictions
2. Throughput - Number of operations per second
3. Scalability - Performance under increasing load
4. Reliability - Success rate and error handling

Usage:
    python tests/performance_test.py --all
    python tests/performance_test.py --scenario light
    python tests/performance_test.py --test response_time
"""

import sys
import os
import argparse
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from surya_forecasting import generator
from pattern_identification import predictor
from tests.test_utils import (
    Timer, PerformanceMetrics, ResourceMonitor, TestResults,
    print_header, print_section, format_duration, format_throughput
)


class PerformanceTestSuite:
    """Main performance testing orchestrator"""
    
    def __init__(self, config_path: str = "tests/test_config.yaml"):
        self.config = self.load_config(config_path)
        self.models = None
        self.results = TestResults("performance_test")
        
    def load_config(self, config_path: str) -> dict:
        """Load test configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup(self):
        """Setup test environment and load models"""
        print_header("Performance Test Suite - Setup")
        
        try:
            # Load models
            model_path = Path(__file__).parent.parent / 'pattern_identification' / 'radio_blackout_models.pkl'
            self.models = predictor.load_models(str(model_path))
            print("✓ Models loaded successfully")
            
            # Add system metadata
            import platform
            self.results.add_metadata('python_version', platform.python_version())
            self.results.add_metadata('platform', platform.platform())
            self.results.add_metadata('processor', platform.processor())
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            raise
    
    def test_response_time(self, iterations: int = 10) -> dict:
        """Test response time for forecast generation and prediction"""
        print_section("Response Time Test")
        
        forecast_times = []
        prediction_times = []
        end_to_end_times = []
        
        # Warmup
        print("Warming up...")
        for _ in range(2):
            _ = generator.generate_24hour_forecast(save_csv=False)
        
        print(f"Running {iterations} iterations...")
        resource_monitor = ResourceMonitor().start()
        
        for i in range(iterations):
            # Measure end-to-end time
            e2e_timer = Timer().start()
            
            # Measure forecast generation
            timer = Timer().start()
            solar_df = generator.generate_24hour_forecast(save_csv=False)
            forecast_time = timer.stop()
            forecast_times.append(forecast_time)
            
            # Measure prediction
            timer = Timer().start()
            hf_df = predictor.generate_hf_blackout_forecast(solar_df, self.models, save_csv=False)
            prediction_time = timer.stop()
            prediction_times.append(prediction_time)
            
            e2e_time = e2e_timer.stop()
            end_to_end_times.append(e2e_time)
            
            resource_monitor.sample()
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{iterations} iterations")
        
        # Calculate statistics
        forecast_stats = PerformanceMetrics.calculate(forecast_times)
        prediction_stats = PerformanceMetrics.calculate(prediction_times)
        e2e_stats = PerformanceMetrics.calculate(end_to_end_times)
        resource_stats = resource_monitor.get_stats()
        
        print("\n📊 Forecast Generation:")
        print(PerformanceMetrics.format_stats(forecast_stats))
        
        print("\n📊 HF Prediction:")
        print(PerformanceMetrics.format_stats(prediction_stats))
        
        print("\n📊 End-to-End:")
        print(PerformanceMetrics.format_stats(e2e_stats))
        
        print("\n💻 Resource Usage:")
        print(f"Peak Memory: {resource_stats['peak_memory_mb']:.2f} MB")
        print(f"Avg CPU: {resource_stats['avg_cpu_percent']:.1f}%")
        
        # Check thresholds
        thresholds = self.config['thresholds']['response_time']
        print("\n✅ Threshold Checks:")
        print(f"Forecast Gen (mean): {forecast_stats['mean']:.2f}s <= {thresholds['forecast_generation_max']}s: "
              f"{'PASS' if forecast_stats['mean'] <= thresholds['forecast_generation_max'] else 'FAIL'}")
        print(f"Prediction (mean): {prediction_stats['mean']:.2f}s <= {thresholds['model_inference_max']}s: "
              f"{'PASS' if prediction_stats['mean'] <= thresholds['model_inference_max'] else 'FAIL'}")
        print(f"E2E (p95): {e2e_stats['p95']:.2f}s <= {thresholds['p95_max']}s: "
              f"{'PASS' if e2e_stats['p95'] <= thresholds['p95_max'] else 'FAIL'}")
        
        return {
            'forecast_generation': forecast_stats,
            'prediction': prediction_stats,
            'end_to_end': e2e_stats,
            'resources': resource_stats,
            'raw_times': {
                'forecast': forecast_times,
                'prediction': prediction_times,
                'e2e': end_to_end_times
            }
        }
    
    def test_throughput(self, duration: int = 30) -> dict:
        """Test throughput - how many forecasts can be generated in a time period"""
        print_section("Throughput Test")
        
        print(f"Generating forecasts for {duration} seconds...")
        
        forecast_count = 0
        prediction_count = 0
        timer = Timer().start()
        
        while timer.elapsed is None or timer.elapsed < duration:
            solar_df = generator.generate_24hour_forecast(save_csv=False)
            forecast_count += 1
            
            _ = predictor.generate_hf_blackout_forecast(solar_df, self.models, save_csv=False)
            prediction_count += 1
            
            # Update elapsed time
            import time
            current_time = time.perf_counter()
            if timer.start_time:
                timer.elapsed = current_time - timer.start_time
        
        total_time = timer.stop()
        
        forecast_rate = forecast_count / total_time
        prediction_rate = prediction_count / total_time
        
        print(f"\n📊 Results:")
        print(f"Duration: {format_duration(total_time)}")
        print(f"Forecasts: {forecast_count} ({format_throughput(forecast_count, total_time)})")
        print(f"Predictions: {prediction_count} ({format_throughput(prediction_count, total_time)})")
        
        # Check thresholds
        thresholds = self.config['thresholds']['throughput']
        print("\n✅ Threshold Checks:")
        print(f"Forecast Rate: {forecast_rate:.2f} >= {thresholds['min_forecasts_per_second']}/s: "
              f"{'PASS' if forecast_rate >= thresholds['min_forecasts_per_second'] else 'FAIL'}")
        
        return {
            'duration': total_time,
            'forecast_count': forecast_count,
            'prediction_count': prediction_count,
            'forecast_rate': forecast_rate,
            'prediction_rate': prediction_rate
        }
    
    def test_scalability(self, max_workers: int = 10) -> dict:
        """Test scalability under concurrent load"""
        print_section("Scalability Test")
        
        baseline_time = None
        results = {}
        
        for num_workers in [1, 2, 5, max_workers]:
            print(f"\nTesting with {num_workers} concurrent worker(s)...")
            
            tasks_per_worker = 5
            total_tasks = num_workers * tasks_per_worker
            
            def run_forecast():
                """Single forecast generation task"""
                solar_df = generator.generate_24hour_forecast(save_csv=False)
                _ = predictor.generate_hf_blackout_forecast(solar_df, self.models, save_csv=False)
                return True
            
            timer = Timer().start()
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(run_forecast) for _ in range(total_tasks)]
                completed = sum(1 for f in as_completed(futures) if f.result())
            
            elapsed = timer.stop()
            
            avg_time_per_task = elapsed / total_tasks
            throughput = total_tasks / elapsed
            
            if baseline_time is None:
                baseline_time = avg_time_per_task
                degradation = 1.0
            else:
                degradation = avg_time_per_task / baseline_time
            
            results[num_workers] = {
                'total_tasks': total_tasks,
                'elapsed': elapsed,
                'avg_time_per_task': avg_time_per_task,
                'throughput': throughput,
                'degradation_factor': degradation
            }
            
            print(f"  Completed: {completed}/{total_tasks}")
            print(f"  Total Time: {format_duration(elapsed)}")
            print(f"  Avg Time/Task: {format_duration(avg_time_per_task)}")
            print(f"  Throughput: {throughput:.2f} tasks/sec")
            print(f"  Degradation: {degradation:.2f}x")
        
        # Check threshold
        max_degradation = results[max_workers]['degradation_factor']
        threshold = self.config['thresholds']['scalability']['max_degradation_factor']
        
        print(f"\n✅ Threshold Check:")
        print(f"Max Degradation: {max_degradation:.2f}x <= {threshold}x: "
              f"{'PASS' if max_degradation <= threshold else 'FAIL'}")
        
        return results
    
    def test_reliability(self, iterations: int = 100) -> dict:
        """Test reliability - success rate and error handling"""
        print_section("Reliability Test")
        
        print(f"Running {iterations} iterations to test reliability...")
        
        successes = 0
        failures = 0
        errors = []
        
        for i in range(iterations):
            try:
                solar_df = generator.generate_24hour_forecast(save_csv=False)
                hf_df = predictor.generate_hf_blackout_forecast(solar_df, self.models, save_csv=False)
                
                # Validate outputs
                assert len(solar_df) == 24, "Solar forecast should have 24 hours"
                assert len(hf_df) == 24, "HF forecast should have 24 hours"
                assert 'hf_blackout_probability' in hf_df.columns, "Missing probability column"
                assert all(0 <= p <= 1 for p in hf_df['hf_blackout_probability']), "Probabilities out of range"
                
                successes += 1
                
            except Exception as e:
                failures += 1
                errors.append(str(e))
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{iterations}")
        
        success_rate = successes / iterations
        error_rate = failures / iterations
        
        print(f"\n📊 Results:")
        print(f"Successes: {successes}/{iterations} ({success_rate*100:.1f}%)")
        print(f"Failures: {failures}/{iterations} ({error_rate*100:.1f}%)")
        
        if errors:
            print(f"\nSample Errors (first 3):")
            for error in errors[:3]:
                print(f"  - {error}")
        
        # Check thresholds
        thresholds = self.config['thresholds']['reliability']
        print(f"\n✅ Threshold Checks:")
        print(f"Success Rate: {success_rate:.2%} >= {thresholds['min_success_rate']:.2%}: "
              f"{'PASS' if success_rate >= thresholds['min_success_rate'] else 'FAIL'}")
        print(f"Error Rate: {error_rate:.2%} <= {thresholds['max_error_rate']:.2%}: "
              f"{'PASS' if error_rate <= thresholds['max_error_rate'] else 'FAIL'}")
        
        return {
            'iterations': iterations,
            'successes': successes,
            'failures': failures,
            'success_rate': success_rate,
            'error_rate': error_rate,
            'errors': errors
        }
    
    def run_scenario(self, scenario_name: str):
        """Run a predefined test scenario"""
        if scenario_name not in self.config['scenarios']:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.config['scenarios'][scenario_name]
        print_header(f"Running Scenario: {scenario['description']}")
        
        # Run tests with scenario parameters
        response_results = self.test_response_time(iterations=scenario['iterations'])
        self.results.add_metric('response_time', response_results)
        
        throughput_results = self.test_throughput(duration=20)
        self.results.add_metric('throughput', throughput_results)
        
        scalability_results = self.test_scalability(max_workers=scenario['concurrent_requests'])
        self.results.add_metric('scalability', scalability_results)
        
        reliability_results = self.test_reliability(iterations=scenario['iterations'])
        self.results.add_metric('reliability', reliability_results)
        
        # Save results
        self.results.save_json()
        self.results.save_summary()
    
    def run_all_tests(self):
        """Run all performance tests with default settings"""
        print_header("Running All Performance Tests")
        
        # Response Time
        response_results = self.test_response_time(iterations=50)
        self.results.add_metric('response_time', response_results)
        
        # Throughput
        throughput_results = self.test_throughput(duration=30)
        self.results.add_metric('throughput', throughput_results)
        
        # Scalability
        scalability_results = self.test_scalability(max_workers=10)
        self.results.add_metric('scalability', scalability_results)
        
        # Reliability
        reliability_results = self.test_reliability(iterations=100)
        self.results.add_metric('reliability', reliability_results)
        
        # Save results
        print_header("Saving Results")
        self.results.save_json()
        self.results.save_summary()
        
        print("\n✨ All tests completed successfully!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Performance Testing Suite')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--scenario', choices=['light', 'medium', 'heavy'], help='Run predefined scenario')
    parser.add_argument('--test', choices=['response_time', 'throughput', 'scalability', 'reliability'],
                       help='Run specific test')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    parser.add_argument('--config', default='tests/test_config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Create test suite
    suite = PerformanceTestSuite(config_path=args.config)
    suite.setup()
    
    try:
        if args.all:
            suite.run_all_tests()
        elif args.scenario:
            suite.run_scenario(args.scenario)
        elif args.test:
            if args.test == 'response_time':
                results = suite.test_response_time(iterations=args.iterations)
            elif args.test == 'throughput':
                results = suite.test_throughput(duration=30)
            elif args.test == 'scalability':
                results = suite.test_scalability(max_workers=10)
            elif args.test == 'reliability':
                results = suite.test_reliability(iterations=args.iterations)
            
            suite.results.add_metric(args.test, results)
            suite.results.save_json()
            suite.results.save_summary()
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
