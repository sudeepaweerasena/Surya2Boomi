# Performance Testing Guide

This guide explains how to use the performance testing framework to evaluate the Surya2Boomi solar flare forecasting system.

## Overview

The testing framework evaluates five key performance attributes:

1. **Response Time** - Latency of forecast generation and predictions
2. **Accuracy** - Model prediction quality and consistency
3. **Throughput** - Number of operations per second
4. **Scalability** - Performance under increasing load
5. **Reliability** - Success rate and error handling

## Installation

Ensure you have the required dependencies:

```bash
pip install pyyaml psutil scikit-learn
```

## Running Tests

### Run All Tests

Execute the complete test suite with default settings:

```bash
cd /Users/sudeepaweerasena/Desktop/Surya2Boomi
python tests/performance_test.py --all
```

### Run Predefined Scenarios

Choose from light, medium, or heavy load scenarios:

```bash
# Light load (10 iterations, 1 concurrent request)
python tests/performance_test.py --scenario light

# Medium load (50 iterations, 5 concurrent requests)  
python tests/performance_test.py --scenario medium

# Heavy load (100 iterations, 10 concurrent requests)
python tests/performance_test.py --scenario heavy
```

### Run Individual Tests

Test specific performance attributes:

```bash
# Response time only
python tests/performance_test.py --test response_time --iterations 50

# Throughput only
python tests/performance_test.py --test throughput

# Scalability only
python tests/performance_test.py --test scalability

# Reliability only
python tests/performance_test.py --test reliability --iterations 100
```

### Run Accuracy Tests

Validate model accuracy with synthetic test data:

```bash
# Run all accuracy tests with 100 samples
python tests/accuracy_test.py --validate --samples 100

# Quick validation with 50 samples
python tests/accuracy_test.py --validate --samples 50
```

## Viewing Results

All test results are automatically saved in `tests/results/` directory:

- **JSON files** (`*.json`) - Complete test data with all metrics
- **Summary files** (`*_summary.txt`) - Human-readable text summaries

### View Results

```bash
# List all test results
ls -lh tests/results/

# View formatted JSON
cat tests/results/performance_test_*.json | python -m json.tool

# View latest text summary
cat tests/results/performance_test_*_summary.txt | tail -50

# View specific result
cat tests/results/performance_test_20260203_163758.json
```

### JSON Structure

The JSON results contain:
- **metrics**: All performance measurements
  - `response_time`: Latency statistics (mean, median, p95, p99)
  - `throughput`: Operations per second
  - `scalability`: Performance under concurrent load
  - `reliability`: Success rates and error tracking
- **metadata**: System information (Python version, platform, processor)
- **timestamp**: When the test was run

## Understanding Metrics

### Response Time

- **Mean** - Average execution time
- **Median** - Middle value (50th percentile)
- **P95** - 95th percentile (95% of requests faster than this)
- **P99** - 99th percentile (99% of requests faster than this)

**Thresholds:**
- Forecast generation: < 10 seconds (mean)
- Model inference: < 2 seconds (mean)
- End-to-end P95: < 15 seconds

### Throughput

- **Forecast Rate** - Forecasts generated per second
- **Prediction Rate** - Predictions made per second

**Thresholds:**
- Minimum: 0.5 forecasts/second

### Scalability

- **Throughput** - Operations per second under load
- **Degradation Factor** - Slowdown compared to baseline (single worker)

**Thresholds:**
- Maximum degradation: 2.0x (at 10 concurrent workers)

### Reliability

- **Success Rate** - Percentage of successful operations
- **Error Rate** - Percentage of failed operations

**Thresholds:**
- Minimum success rate: 95%
- Maximum error rate: 5%

### Accuracy

- **Probability Consistency** - Probabilities within valid range [0, 1]
- **Severity Classification** - Alignment between severity levels and probabilities
- **Temporal Consistency** - Smooth transitions between hours
- **Confidence Scores** - Correlation with prediction quality

**Metrics:**
- Severity-probability consistency rate
- Temporal smoothness rate
- Correlation with solar flare probability

## Configuration

Customize test parameters in `tests/test_config.yaml`:

```yaml
scenarios:
  custom:
    iterations: 25
    concurrent_requests: 3
    warmup_iterations: 3

thresholds:
  response_time:
    forecast_generation_max: 5.0  # Adjust based on requirements
```

## Interpreting Results

### ✅ PASS Criteria

Tests pass when:
- Response times are below thresholds
- Success rate ≥ 95%
- Scalability degradation ≤ 2.0x
- Probabilities are well-calibrated and consistent

### ❌ FAIL Indicators

Investigate if:
- Response times exceed 2x the threshold
- Success rate drops below 90%
- Large hour-to-hour probability jumps (>0.5)
- High memory usage (>2GB)

## Troubleshooting

### Out of Memory

If tests fail with memory errors:
1. Reduce concurrent workers in scalability tests
2. Decrease number of iterations
3. Monitor system resources during tests

### Slow Performance

If response times are unexpectedly high:
1. Ensure no other heavy processes are running
2. Check if models are properly cached
3. Run warm-up iterations before measurement

### Inconsistent Results

If results vary significantly between runs:
1. Increase number of iterations for better statistical confidence
2. Run tests when system load is stable
3. Check for background processes affecting performance

## Best Practices

1. **Establish Baselines** - Run tests after major changes to track regression
2. **Regular Testing** - Schedule periodic performance tests
3. **Compare Results** - Review JSON files to compare test runs over time
4. **Document Changes** - Note any code changes that affect performance
5. **Optimize** - Focus on bottlenecks identified by tests

## Example Workflow

```bash
# 1. Run comprehensive test suite
python tests/performance_test.py --all

# 2. Run accuracy validation  
python tests/accuracy_test.py --validate

# 3. View results
cat tests/results/performance_test_*_summary.txt

# 4. For CI/CD, run quick validation
python tests/performance_test.py --scenario light
```

## Performance Optimization Tips

Based on test results, consider these optimizations:

- **High Response Time** - Profile code to find bottlenecks, optimize model inference
- **Low Throughput** - Implement batch processing, use async operations
- **Poor Scalability** - Check for threading issues, add caching
- **Low Reliability** - Add error handling, input validation
- **Accuracy Issues** - Retrain models, adjust probability thresholds

## Contact & Support

For questions about performance testing:
- Check the configuration in `test_config.yaml`
- Review test implementation in `performance_test.py`
- Examine utility functions in `test_utils.py`
