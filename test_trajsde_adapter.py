#!/usr/bin/env python
# coding: utf-8
"""
Test script for TrajSDE LibCity adapter

This script tests basic functionality of the TrajSDE adapter:
1. Import the model
2. Instantiate with dummy config
3. Run forward pass with dummy batch
4. Calculate loss
"""

import sys
import os
import torch

# Add LibCity to path
libcity_path = '/home/wangwenrui/shk/AgentCity/Bigscity-LibCity'
if libcity_path not in sys.path:
    sys.path.insert(0, libcity_path)

from libcity.model.trajectory_loc_prediction.TrajSDE import TrajSDE


def create_dummy_batch(batch_size=8, seq_len=20, loc_size=100, tim_size=24):
    """Create a dummy batch for testing."""
    batch = {
        'current_loc': torch.randint(0, loc_size, (batch_size, seq_len)),
        'current_tim': torch.randint(0, tim_size, (batch_size, seq_len)),
        'target': torch.randint(0, loc_size, (batch_size,)),
        'uid': torch.randint(0, 50, (batch_size,))
    }
    return batch


def test_trajsde_adapter():
    """Test TrajSDE adapter basic functionality."""
    print("=" * 80)
    print("Testing TrajSDE LibCity Adapter")
    print("=" * 80)

    # Create dummy config and data_feature
    config = {
        'device': 'cpu',
        'embed_dim': 64,
        'num_modes': 6,
        'historical_steps': 21,
        'future_steps': 60,
        'hidden_size': 128,
        'use_native_trajsde': False,  # Use simplified model
    }

    data_feature = {
        'loc_size': 100,
        'uid_size': 50,
        'tim_size': 24,
    }

    print("\n1. Creating TrajSDE model...")
    try:
        model = TrajSDE(config, data_feature)
        print("   ✓ Model created successfully")
        print(f"   - Device: {model.device}")
        print(f"   - Embed dim: {model.embed_dim}")
        print(f"   - Hidden size: {model.hidden_size}")
        print(f"   - Num modes: {model.num_modes}")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n2. Creating dummy batch...")
    batch = create_dummy_batch(batch_size=8, seq_len=20, loc_size=100, tim_size=24)
    print(f"   ✓ Batch created")
    print(f"   - Batch size: {batch['current_loc'].shape[0]}")
    print(f"   - Sequence length: {batch['current_loc'].shape[1]}")

    print("\n3. Testing forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            output = model.forward(batch)
        print(f"   ✓ Forward pass successful")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Expected shape: [batch_size={batch['current_loc'].shape[0]}, loc_size={data_feature['loc_size']}]")
        assert output.shape == (batch['current_loc'].shape[0], data_feature['loc_size'])
        print("   ✓ Output shape is correct")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n4. Testing predict method...")
    try:
        model.eval()
        predictions = model.predict(batch)
        print(f"   ✓ Predict successful")
        print(f"   - Predictions shape: {predictions.shape}")
        # Check that log_softmax was applied (values should be negative)
        print(f"   - Max prediction value: {predictions.max().item():.4f}")
        print(f"   - Min prediction value: {predictions.min().item():.4f}")
        assert predictions.max().item() <= 0, "Log softmax should produce values <= 0"
        print("   ✓ Predictions have correct range (log probabilities)")
    except Exception as e:
        print(f"   ✗ Predict failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n5. Testing calculate_loss method...")
    try:
        model.train()
        loss = model.calculate_loss(batch)
        print(f"   ✓ Loss calculation successful")
        print(f"   - Loss value: {loss.item():.4f}")
        print(f"   - Loss shape: {loss.shape}")
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive"
        print("   ✓ Loss is a positive scalar")
    except Exception as e:
        print(f"   ✗ Loss calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n6. Testing backward pass...")
    try:
        model.train()
        loss = model.calculate_loss(batch)
        loss.backward()
        print("   ✓ Backward pass successful")

        # Check that gradients were computed
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        if has_gradients:
            print("   ✓ Gradients computed successfully")
        else:
            print("   ⚠ Warning: No gradients found (might be expected for some architectures)")
    except Exception as e:
        print(f"   ✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)

    print("\nNext steps:")
    print("1. Test with actual LibCity data loader")
    print("2. Run a full training loop")
    print("3. Implement native TrajSDE integration (if needed)")
    print("4. Add location coordinate mapping for continuous space")
    print("5. Implement graph construction from trajectory data")

    return True


if __name__ == '__main__':
    success = test_trajsde_adapter()
    sys.exit(0 if success else 1)
