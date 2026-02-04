"""
DeepMMEvaluator: Evaluator for DeepMM seq2seq map matching model.

This evaluator is designed for sequence-to-sequence map matching models like DeepMM
that produce road segment sequences as output. It works with the batch format from
DeepMMSeq2SeqDataset.

Unlike MapMatchingEvaluator which requires road network data ('rd_nwk') for
computing road network-based metrics (RMF, AN, AL), this evaluator works with
pure sequence matching metrics suitable for seq2seq models.

Metrics implemented:
- Accuracy: Percentage of correctly predicted road segments (after removing
            consecutive duplicates to handle repeated segments)
- Edit Distance / Levenshtein Distance: Measures the minimum edits needed to
            transform prediction into ground truth
- Sequence Accuracy: Percentage of sequences with exact match

Adapted from the original DeepMM evaluation methodology.

Batch Format expected:
    {
        'result': torch.LongTensor,  # Predicted road segments [batch_size, seq_len]
        'batch': {
            'output_trg': torch.LongTensor,  # Ground truth segments [batch_size, seq_len]
            'target': torch.LongTensor,       # Alias for output_trg (same content)
        }
    }
"""

import os
import json
import datetime
import numpy as np
import torch
from logging import getLogger
from collections import defaultdict

from libcity.evaluator.abstract_evaluator import AbstractEvaluator
from libcity.utils import ensure_dir


def remove_consecutive_duplicates(seq, pad_id=1):
    """Remove consecutive duplicate tokens from a sequence.

    Also removes padding tokens. This is important for map matching evaluation
    because the model may predict the same road segment multiple times for
    consecutive GPS points on the same road.

    Args:
        seq: List or tensor of token IDs
        pad_id: ID of padding token to exclude

    Returns:
        List of unique consecutive tokens (no padding)
    """
    if isinstance(seq, torch.Tensor):
        seq = seq.cpu().numpy().tolist()
    elif isinstance(seq, np.ndarray):
        seq = seq.tolist()

    result = []
    prev_token = None
    for token in seq:
        if token != pad_id and token != prev_token:
            result.append(token)
            prev_token = token
    return result


def levenshtein_distance(seq1, seq2):
    """Compute Levenshtein (edit) distance between two sequences.

    Args:
        seq1: First sequence (list of tokens)
        seq2: Second sequence (list of tokens)

    Returns:
        int: Minimum number of insertions, deletions, substitutions
             required to transform seq1 into seq2
    """
    len1, len2 = len(seq1), len(seq2)

    # Create distance matrix
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Initialize base cases
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # Fill in the matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )

    return dp[len1][len2]


def longest_common_subsequence(seq1, seq2):
    """Compute length of longest common subsequence between two sequences.

    Args:
        seq1: First sequence (list of tokens)
        seq2: Second sequence (list of tokens)

    Returns:
        int: Length of LCS
    """
    len1, len2 = len(seq1), len(seq2)

    # Create LCS matrix
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Fill in the matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[len1][len2]


class DeepMMEvaluator(AbstractEvaluator):
    """Evaluator for DeepMM seq2seq map matching model.

    This evaluator is designed for sequence-to-sequence map matching models
    that produce road segment sequences. It uses sequence-level metrics instead
    of road network-based metrics.
    """

    def __init__(self, config):
        """Initialize the DeepMM evaluator.

        Args:
            config: Configuration dictionary containing:
                - metrics: List of metrics to compute. Allowed values:
                    'accuracy', 'edit_distance', 'lcs', 'seq_accuracy'
                - pad_token: Padding token ID (default: 1)
                - sos_token: Start-of-sequence token ID (default: 0)
                - eos_token: End-of-sequence token ID (default: 2)
                - save_modes: List of save formats ['csv', 'json']
        """
        self.config = config
        self.metrics = config.get('metrics', ['accuracy', 'edit_distance'])
        self.allowed_metrics = ['accuracy', 'edit_distance', 'lcs', 'seq_accuracy',
                                'normalized_edit_distance', 'lcs_ratio']
        self.save_modes = config.get('save_modes', ['csv', 'json'])
        self._logger = getLogger()

        # Special token IDs
        self.pad_token = config.get('pad_token', 1)
        self.sos_token = config.get('sos_token', 0)
        self.eos_token = config.get('eos_token', 2)

        # Intermediate results storage
        self.intermediate_result = defaultdict(list)
        self.evaluate_result = {}

        self._check_config()

    def _check_config(self):
        """Validate configuration."""
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator metrics should be a list')
        for metric in self.metrics:
            if metric not in self.allowed_metrics:
                self._logger.warning(
                    'Metric {} not in allowed metrics {}. Will be ignored.'.format(
                        metric, self.allowed_metrics))

    def collect(self, batch):
        """Collect predictions and ground truth from a batch for evaluation.

        This method is called by the executor after each batch during validation
        or evaluation. It computes per-sample metrics and stores them for later
        aggregation.

        Args:
            batch: Dictionary containing:
                - 'result': Predicted road segments [batch_size, seq_len] (tensor or list)
                - 'batch': Original batch dictionary containing:
                    - 'output_trg' or 'target': Ground truth segments [batch_size, seq_len]
        """
        if not isinstance(batch, dict):
            raise TypeError('DeepMMEvaluator.collect expects a dict, got {}'.format(type(batch)))

        result = batch.get('result')
        original_batch = batch.get('batch', {})

        # Get ground truth from original batch
        ground_truth = original_batch.get('output_trg')
        if ground_truth is None:
            ground_truth = original_batch.get('target')
        if ground_truth is None:
            self._logger.warning('No ground truth found in batch. Skipping evaluation.')
            return

        # Convert to numpy if tensors
        if isinstance(result, torch.Tensor):
            result = result.cpu().numpy()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.cpu().numpy()

        # Handle different result formats
        # If result is a single sequence (not batched), wrap it
        if len(result.shape) == 1:
            result = result.reshape(1, -1)
        if len(ground_truth.shape) == 1:
            ground_truth = ground_truth.reshape(1, -1)

        batch_size = min(len(result), len(ground_truth))

        for i in range(batch_size):
            # Get prediction and ground truth for this sample
            pred_seq = result[i]
            true_seq = ground_truth[i]

            # Remove consecutive duplicates and padding
            pred_clean = remove_consecutive_duplicates(pred_seq, self.pad_token)
            true_clean = remove_consecutive_duplicates(true_seq, self.pad_token)

            # Remove special tokens (SOS, EOS) if present
            pred_clean = [t for t in pred_clean if t not in [self.sos_token, self.eos_token]]
            true_clean = [t for t in true_clean if t not in [self.sos_token, self.eos_token]]

            # Calculate metrics for this sample
            if len(true_clean) > 0:
                # Token-level accuracy
                # Count how many predicted tokens are correct
                correct = 0
                min_len = min(len(pred_clean), len(true_clean))
                for j in range(min_len):
                    if pred_clean[j] == true_clean[j]:
                        correct += 1

                accuracy = correct / len(true_clean)
                self.intermediate_result['accuracy'].append(accuracy)

                # Edit distance
                edit_dist = levenshtein_distance(pred_clean, true_clean)
                self.intermediate_result['edit_distance'].append(edit_dist)

                # Normalized edit distance (by ground truth length)
                norm_edit_dist = edit_dist / max(len(true_clean), 1)
                self.intermediate_result['normalized_edit_distance'].append(norm_edit_dist)

                # Longest common subsequence
                lcs_len = longest_common_subsequence(pred_clean, true_clean)
                self.intermediate_result['lcs'].append(lcs_len)

                # LCS ratio (by ground truth length)
                lcs_ratio = lcs_len / max(len(true_clean), 1)
                self.intermediate_result['lcs_ratio'].append(lcs_ratio)

                # Sequence accuracy (exact match)
                seq_match = 1 if pred_clean == true_clean else 0
                self.intermediate_result['seq_accuracy'].append(seq_match)

    def evaluate(self):
        """Compute final evaluation metrics from collected results.

        This method aggregates the per-sample metrics collected during batch
        processing and computes the final evaluation results.

        Returns:
            dict: Dictionary of metric names to values
        """
        self.evaluate_result = {}

        # Compute mean of each metric
        for metric in self.allowed_metrics:
            if metric in self.intermediate_result and len(self.intermediate_result[metric]) > 0:
                values = self.intermediate_result[metric]
                if metric in ['accuracy', 'lcs_ratio', 'seq_accuracy']:
                    # These are already ratios, just average them
                    self.evaluate_result[metric] = np.mean(values)
                elif metric == 'edit_distance':
                    # Report mean edit distance
                    self.evaluate_result[metric] = np.mean(values)
                elif metric == 'normalized_edit_distance':
                    # Report mean normalized edit distance
                    self.evaluate_result[metric] = np.mean(values)
                elif metric == 'lcs':
                    # Report mean LCS length
                    self.evaluate_result[metric] = np.mean(values)

        # Add sample count
        self.evaluate_result['num_samples'] = len(self.intermediate_result.get('accuracy', []))

        return self.evaluate_result

    def save_result(self, save_path, filename=None):
        """Save evaluation results to files.

        Args:
            save_path: Directory to save results
            filename: Optional filename prefix. If None, uses timestamp.
        """
        # Ensure results are computed
        self.evaluate()

        ensure_dir(save_path)

        if filename is None:
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                       self.config.get('model', 'DeepMM') + '_' + \
                       self.config.get('dataset', 'unknown')

        # Log results
        self._logger.info('DeepMM Evaluation Results: {}'.format(
            json.dumps(self.evaluate_result, indent=2)))

        # Save as JSON
        if 'json' in self.save_modes:
            json_path = os.path.join(save_path, '{}.json'.format(filename))
            with open(json_path, 'w') as f:
                json.dump(self.evaluate_result, f, indent=4)
            self._logger.info('Evaluation results saved to {}'.format(json_path))

        # Save as CSV
        if 'csv' in self.save_modes:
            csv_path = os.path.join(save_path, '{}.csv'.format(filename))
            with open(csv_path, 'w') as f:
                # Write header
                headers = list(self.evaluate_result.keys())
                f.write(','.join(headers) + '\n')
                # Write values
                values = [str(self.evaluate_result[h]) for h in headers]
                f.write(','.join(values) + '\n')
            self._logger.info('Evaluation results saved to {}'.format(csv_path))

    def clear(self):
        """Clear collected intermediate results.

        This should be called at the start of each evaluation epoch to reset
        the accumulated metrics.
        """
        self.intermediate_result.clear()
        self.evaluate_result = {}
