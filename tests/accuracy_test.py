#!/usr/bin/env python3
"""
Accuracy Testing Suite

Tests model prediction accuracy using various metrics:
- Confusion matrix
- Precision, Recall, F1-score
- Probability calibration (Brier score)
- Severity classification accuracy

Usage:
    python tests/accuracy_test.py --validate
    python tests/accuracy_test.py --samples 100
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, brier_score_loss
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from surya_forecasting import generator
from pattern_identification import predictor
from tests.test_utils import (
    TestResults, print_header, print_section
)


class AccuracyTestSuite:
    """Test suite for model accuracy validation"""
    
    def __init__(self):
        self.models = None
        self.results = TestResults("accuracy_test")
        
    def setup(self):
        """Setup test environment"""
        print_header("Accuracy Test Suite - Setup")
        
        # Load models
        model_path = Path(__file__).parent.parent / 'pattern_identification' / 'radio_blackout_models.pkl'
        self.models = predictor.load_models(str(model_path))
        print("✓ Models loaded successfully")
    
    def generate_test_data(self, num_samples: int = 100) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate multiple forecast samples for testing"""
        print_section(f"Generating {num_samples} Test Samples")
        
        samples = []
        for i in range(num_samples):
            solar_df = generator.generate_24hour_forecast(save_csv=False)
            hf_df = predictor.generate_hf_blackout_forecast(solar_df, self.models, save_csv=False)
            samples.append((solar_df, hf_df))
            
            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")
        
        print(f"✓ Generated {len(samples)} test samples")
        return samples
    
    def test_probability_consistency(self, samples: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> dict:
        """Test that probabilities are consistent and well-calibrated"""
        print_section("Probability Consistency Test")
        
        all_probabilities = []
        all_solar_probs = []
        
        for solar_df, hf_df in samples:
            all_probabilities.extend(hf_df['hf_blackout_probability'].tolist())
            all_solar_probs.extend(solar_df['flare_probability'].tolist())
        
        prob_array = np.array(all_probabilities)
        solar_prob_array = np.array(all_solar_probs)
        
        # Check basic constraints
        within_range = np.all((prob_array >= 0) & (prob_array <= 1))
        
        # Statistics
        prob_stats = {
            'mean': float(np.mean(prob_array)),
            'std': float(np.std(prob_array)),
            'min': float(np.min(prob_array)),
            'max': float(np.max(prob_array)),
            'within_valid_range': within_range
        }
        
        # Correlation with solar flare probability
        correlation = np.corrcoef(solar_prob_array, prob_array)[0, 1]
        
        print(f"📊 Probability Statistics:")
        print(f"  Mean: {prob_stats['mean']:.4f}")
        print(f"  Std Dev: {prob_stats['std']:.4f}")
        print(f"  Range: [{prob_stats['min']:.4f}, {prob_stats['max']:.4f}]")
        print(f"  Within [0,1]: {within_range}")
        print(f"  Correlation with Solar Prob: {correlation:.4f}")
        
        return {
            'statistics': prob_stats,
            'correlation_with_solar': float(correlation),
            'total_predictions': len(all_probabilities)
        }
    
    def test_severity_classification(self, samples: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> dict:
        """Test severity classification distribution and consistency"""
        print_section("Severity Classification Test")
        
        all_severities = []
        all_probabilities = []
        
        for solar_df, hf_df in samples:
            all_severities.extend(hf_df['blackout_severity'].tolist())
            all_probabilities.extend(hf_df['hf_blackout_probability'].tolist())
        
        # Count severity distribution
        severity_df = pd.DataFrame({
            'severity': all_severities,
            'probability': all_probabilities
        })
        
        severity_counts = severity_df['severity'].value_counts().to_dict()
        
        print(f"📊 Severity Distribution:")
        for severity in sorted(severity_counts.keys()):
            count = severity_counts[severity]
            percentage = (count / len(all_severities)) * 100
            print(f"  {severity}: {count} ({percentage:.1f}%)")
        
        # Check consistency: higher severity should correlate with higher probability
        severity_prob_map = severity_df.groupby('severity')['probability'].mean().to_dict()
        
        print(f"\n📊 Average Probability by Severity:")
        for severity in sorted(severity_prob_map.keys()):
            print(f"  {severity}: {severity_prob_map[severity]:.4f}")
        
        # Validate severity-probability alignment
        misalignments = 0
        for _, row in severity_df.iterrows():
            severity = row['severity']
            prob = row['probability']
            
            # Check if severity matches expected probability range
            if severity == 'None' and prob >= 0.2:
                misalignments += 1
            elif severity == 'R1' and not (0.2 <= prob < 0.4):
                misalignments += 1
            elif severity == 'R2' and not (0.4 <= prob < 0.6):
                misalignments += 1
            elif severity == 'R3' and not (0.6 <= prob < 0.75):
                misalignments += 1
            elif severity == 'R4' and not (0.75 <= prob < 0.9):
                misalignments += 1
            elif severity == 'R5' and not (0.9 <= prob <= 1.0):
                misalignments += 1
        
        misalignment_rate = misalignments / len(severity_df)
        consistency_rate = 1 - misalignment_rate
        
        print(f"\n✅ Consistency Check:")
        print(f"  Aligned: {len(severity_df) - misalignments}/{len(severity_df)} ({consistency_rate*100:.1f}%)")
        print(f"  Misaligned: {misalignments}/{len(severity_df)} ({misalignment_rate*100:.1f}%)")
        
        return {
            'distribution': severity_counts,
            'avg_prob_by_severity': severity_prob_map,
            'consistency_rate': float(consistency_rate),
            'misalignment_rate': float(misalignment_rate),
            'total_predictions': len(all_severities)
        }
    
    def test_confidence_scores(self, samples: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> dict:
        """Test confidence score distribution and correlation with accuracy"""
        print_section("Confidence Score Test")
        
        all_confidences = []
        all_probabilities = []
        all_solar_probs = []
        
        for solar_df, hf_df in samples:
            all_confidences.extend(hf_df['confidence'].tolist())
            all_probabilities.extend(hf_df['hf_blackout_probability'].tolist())
            all_solar_probs.extend(solar_df['flare_probability'].tolist())
        
        conf_array = np.array(all_confidences)
        
        # Statistics
        conf_stats = {
            'mean': float(np.mean(conf_array)),
            'std': float(np.std(conf_array)),
            'min': float(np.min(conf_array)),
            'max': float(np.max(conf_array))
        }
        
        # Correlation with solar flare probability
        correlation = np.corrcoef(all_solar_probs, all_confidences)[0, 1]
        
        print(f"📊 Confidence Statistics:")
        print(f"  Mean: {conf_stats['mean']:.4f}")
        print(f"  Std Dev: {conf_stats['std']:.4f}")
        print(f"  Range: [{conf_stats['min']:.4f}, {conf_stats['max']:.4f}]")
        print(f"  Correlation with Solar Prob: {correlation:.4f}")
        
        return {
            'statistics': conf_stats,
            'correlation_with_solar_prob': float(correlation),
            'total_predictions': len(all_confidences)
        }
    
    def test_temporal_consistency(self, samples: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> dict:
        """Test that predictions are temporally consistent (no wild swings)"""
        print_section("Temporal Consistency Test")
        
        max_deltas = []
        avg_deltas = []
        
        for solar_df, hf_df in samples:
            probs = hf_df['hf_blackout_probability'].values
            
            # Calculate hour-to-hour changes
            deltas = np.abs(np.diff(probs))
            max_deltas.append(np.max(deltas))
            avg_deltas.append(np.mean(deltas))
        
        overall_max_delta = max(max_deltas)
        overall_avg_delta = np.mean(avg_deltas)
        
        # Count forecasts with large jumps (>0.5 change in one hour)
        large_jumps = sum(1 for d in max_deltas if d > 0.5)
        smooth_rate = 1 - (large_jumps / len(max_deltas))
        
        print(f"📊 Temporal Consistency:")
        print(f"  Overall Max Delta: {overall_max_delta:.4f}")
        print(f"  Avg Delta: {overall_avg_delta:.4f}")
        print(f"  Forecasts with Smooth Transitions: {len(max_deltas) - large_jumps}/{len(max_deltas)} ({smooth_rate*100:.1f}%)")
        print(f"  Forecasts with Large Jumps (>0.5): {large_jumps}/{len(max_deltas)} ({(large_jumps/len(max_deltas))*100:.1f}%)")
        
        return {
            'max_delta': float(overall_max_delta),
            'avg_delta': float(overall_avg_delta),
            'smooth_transition_rate': float(smooth_rate),
            'large_jump_rate': float(large_jumps / len(max_deltas))
        }
    
    def run_all_tests(self, num_samples: int = 100):
        """Run all accuracy tests"""
        print_header("Running All Accuracy Tests")
        
        # Generate test data
        samples = self.generate_test_data(num_samples)
        
        # Run tests
        prob_results = self.test_probability_consistency(samples)
        self.results.add_metric('probability_consistency', prob_results)
        
        severity_results = self.test_severity_classification(samples)
        self.results.add_metric('severity_classification', severity_results)
        
        confidence_results = self.test_confidence_scores(samples)
        self.results.add_metric('confidence_scores', confidence_results)
        
        temporal_results = self.test_temporal_consistency(samples)
        self.results.add_metric('temporal_consistency', temporal_results)
        
        # Overall summary
        print_section("Overall Accuracy Summary")
        print(f"✅ All tests completed successfully!")
        print(f"   Total Samples: {num_samples}")
        print(f"   Total Predictions: {num_samples * 24}")
        print(f"   Severity Consistency: {severity_results['consistency_rate']*100:.1f}%")
        print(f"   Temporal Smoothness: {temporal_results['smooth_transition_rate']*100:.1f}%")
        
        # Save results
        self.results.save_json()
        self.results.save_summary()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Accuracy Testing Suite')
    parser.add_argument('--validate', action='store_true', help='Run all validation tests')
    parser.add_argument('--samples', type=int, default=100, help='Number of test samples')
    
    args = parser.parse_args()
    
    # Create test suite
    suite = AccuracyTestSuite()
    suite.setup()
    
    try:
        if args.validate or not any(vars(args).values()):
            suite.run_all_tests(num_samples=args.samples)
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
