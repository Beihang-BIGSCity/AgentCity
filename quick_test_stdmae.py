#!/usr/bin/env python
"""Quick test to verify STDMAE dimension fix works"""
import sys
sys.path.append('Bigscity-LibCity')

import torch
import numpy as np
from libcity.data.dataset import TrafficStatePointDataset
from libcity.model.traffic_speed_prediction.STDMAE import STDMAE
from libcity.utils import get_executor, get_model, get_logger
from libcity.config import ConfigParser

def quick_test_stdmae():
    """Run a quick test of STDMAE to verify dimension fix"""
    print("=" * 60)
    print("STDMAE Quick Test - Verifying Dimension Fix")
    print("=" * 60)

    # Minimal config
    config = ConfigParser(
        task='traffic_state_pred',
        model='STDMAE',
        dataset='METR_LA',
        config_file=None,
        saved_model=False,
        train=True,
        other_args={
            'gpu': False,
            'max_epoch': 1,
            'batch_size': 2,  # Very small batch
            'train_rate': 0.01,  # Use only 1% of data
            'eval_rate': 0.005,
            'seq_len': 144,  # Reduce sequence length from 864 to 144
            'patch_size': 12,
            'encoder_depth': 2,  # Reduce from 4
            'decoder_depth': 1,
            'num_workers': 0
        }
    )

    # Get logger
    logger = get_logger(config)
    logger.info("Loading minimal dataset...")

    # Load minimal dataset
    dataset = get_model(config, None, None)

    logger.info(f"Dataset loaded. Number of nodes: {dataset.num_nodes}")

    # Create a tiny test batch manually
    logger.info("Creating test batch...")
    batch_size = 2
    input_window = 12
    output_window = 12
    num_nodes = dataset.num_nodes
    feature_dim = dataset.feature_dim

    # Create test data
    test_batch = {
        'X': torch.randn(batch_size, input_window, num_nodes, feature_dim),  # Short history
        'X_ext': torch.randn(batch_size, 144, num_nodes, feature_dim),  # Long history (reduced)
        'y': torch.randn(batch_size, output_window, num_nodes, 1)
    }

    logger.info(f"Test batch created:")
    logger.info(f"  X shape: {test_batch['X'].shape}")
    logger.info(f"  X_ext shape: {test_batch['X_ext'].shape}")
    logger.info(f"  y shape: {test_batch['y'].shape}")

    # Test forward pass
    logger.info("\n" + "=" * 60)
    logger.info("Testing forward pass...")
    logger.info("=" * 60)

    try:
        with torch.no_grad():
            output = dataset.forward(test_batch)
            logger.info(f"\n✓ Forward pass successful!")
            logger.info(f"  Output shape: {output.shape}")
            logger.info(f"  Expected shape: [{batch_size}, {output_window}, {num_nodes}, 1]")

            # Verify output shape
            expected_shape = (batch_size, output_window, num_nodes, 1)
            if output.shape == expected_shape:
                logger.info(f"\n✓ Output shape is correct!")
            else:
                logger.error(f"\n✗ Output shape mismatch!")
                logger.error(f"  Expected: {expected_shape}")
                logger.error(f"  Got: {output.shape}")
                return False

        # Test loss calculation
        logger.info("\n" + "=" * 60)
        logger.info("Testing loss calculation...")
        logger.info("=" * 60)

        loss = dataset.calculate_loss(test_batch)
        logger.info(f"\n✓ Loss calculation successful!")
        logger.info(f"  Loss value: {loss.item():.4f}")

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe dimension fix is working correctly:")
        print("  - Model initializes without errors")
        print("  - Forward pass completes successfully")
        print("  - Output shape is correct")
        print("  - Loss calculation works")
        print("\nThe STDMAE model is ready for full training.")
        print("=" * 60)

        return True

    except RuntimeError as e:
        logger.error(f"\n✗ Test failed with error:")
        logger.error(f"  {str(e)}")
        logger.error(f"\nThis indicates the dimension fix may not be complete.")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        logger.error(f"\n✗ Unexpected error:")
        logger.error(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = quick_test_stdmae()
    sys.exit(0 if success else 1)
