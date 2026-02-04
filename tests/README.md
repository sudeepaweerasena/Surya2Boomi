# Performance Testing Quick Start

## Installation

```bash
# Install dependencies
pip install pyyaml psutil scikit-learn
```

## Quick Usage

### 1. Run All Tests (Comprehensive)
```bash
python tests/performance_test.py --all
```

### 2. Run Specific Test
```bash
# Response time (fast)
python tests/performance_test.py --test response_time --iterations 10

# Throughput
python tests/performance_test.py --test throughput

# Scalability  
python tests/performance_test.py --test scalability

# Reliability
python tests/performance_test.py --test reliability
```

### 3. Run Accuracy Tests
```bash
python tests/accuracy_test.py --validate --samples 50
```

## Test Scenarios

```bash
# Light load (quick validation)
python tests/performance_test.py --scenario light

# Medium load
python tests/performance_test.py --scenario medium

# Heavy load (comprehensive)
python tests/performance_test.py --scenario heavy
```

## Viewing Results

All test results are automatically saved as JSON and text summaries:

- **JSON Results**: `tests/results/*.json` - Complete detailed metrics
- **Text Summaries**: `tests/results/*_summary.txt` - Human-readable summaries

### View Latest Results
```bash
# List all results
ls -lh tests/results/

# View JSON (formatted)
cat tests/results/performance_test_*.json | python -m json.tool

# View text summary
cat tests/results/performance_test_*_summary.txt
```

## For More Details

See [PERFORMANCE_TESTING.md](../docs/PERFORMANCE_TESTING.md) for comprehensive documentation.
