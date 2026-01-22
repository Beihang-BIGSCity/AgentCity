"""
Quick test to verify DSTAGNN model adaptation works correctly
"""

import torch
import numpy as np
import sys
sys.path.append('./Bigscity-LibCity')

def test_dstagnn_import():
    """Test if DSTAGNN can be imported"""
    try:
        from libcity.model.traffic_flow_prediction import DSTAGNN
        print("✓ DSTAGNN import successful")
        return True
    except Exception as e:
        print(f"✗ DSTAGNN import failed: {e}")
        return False


def test_dstagnn_initialization():
    """Test if DSTAGNN can be initialized with dummy config"""
    try:
        from libcity.model.traffic_flow_prediction import DSTAGNN

        # Create dummy configuration
        class DummyConfig:
            def get(self, key, default=None):
                config_dict = {
                    'device': torch.device('cpu'),
                    'input_window': 12,
                    'output_window': 12,
                    'in_channels': 1,
                    'nb_block': 2,  # Use 2 blocks for faster testing
                    'K': 3,
                    'nb_chev_filter': 16,  # Reduced for testing
                    'nb_time_filter': 16,  # Reduced for testing
                    'd_model': 64,  # Reduced for testing
                    'd_k': 16,  # Reduced for testing
                    'n_heads': 2,
                    'graph_use': 'AG',
                }
                return config_dict.get(key, default)

        # Create dummy data feature
        num_nodes = 10  # Small graph for testing
        adj_mx = np.random.rand(num_nodes, num_nodes)
        adj_mx = (adj_mx + adj_mx.T) / 2  # Make symmetric

        class DummyScaler:
            def inverse_transform(self, data):
                return data

        class DummyDataFeature:
            def get(self, key, default=None):
                feature_dict = {
                    'num_nodes': num_nodes,
                    'adj_mx': adj_mx,
                    'adj_TMD': adj_mx,
                    'adj_pa': adj_mx,
                    'scaler': DummyScaler(),
                }
                return feature_dict.get(key, default)

        config = DummyConfig()
        data_feature = DummyDataFeature()

        # Initialize model
        model = DSTAGNN(config, data_feature)
        print(f"✓ DSTAGNN initialization successful")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True

    except Exception as e:
        print(f"✗ DSTAGNN initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dstagnn_forward():
    """Test if DSTAGNN forward pass works"""
    try:
        from libcity.model.traffic_flow_prediction import DSTAGNN

        # Setup (same as initialization test)
        class DummyConfig:
            def get(self, key, default=None):
                config_dict = {
                    'device': torch.device('cpu'),
                    'input_window': 12,
                    'output_window': 12,
                    'in_channels': 1,
                    'nb_block': 2,
                    'K': 3,
                    'nb_chev_filter': 16,
                    'nb_time_filter': 16,
                    'd_model': 64,
                    'd_k': 16,
                    'n_heads': 2,
                    'graph_use': 'AG',
                }
                return config_dict.get(key, default)

        num_nodes = 10
        adj_mx = np.random.rand(num_nodes, num_nodes)
        adj_mx = (adj_mx + adj_mx.T) / 2

        class DummyScaler:
            def inverse_transform(self, data):
                return data

        class DummyDataFeature:
            def get(self, key, default=None):
                feature_dict = {
                    'num_nodes': num_nodes,
                    'adj_mx': adj_mx,
                    'adj_TMD': adj_mx,
                    'adj_pa': adj_mx,
                    'scaler': DummyScaler(),
                }
                return feature_dict.get(key, default)

        config = DummyConfig()
        data_feature = DummyDataFeature()
        model = DSTAGNN(config, data_feature)

        # Create dummy batch
        batch_size = 4
        input_window = 12
        output_window = 12

        batch = {
            'X': torch.randn(batch_size, input_window, num_nodes, 1),
            'y': torch.randn(batch_size, output_window, num_nodes, 1),
        }

        # Test forward pass
        with torch.no_grad():
            output = model.predict(batch)

        expected_shape = (batch_size, output_window, num_nodes, 1)
        assert output.shape == expected_shape, f"Output shape {output.shape} != expected {expected_shape}"

        print(f"✓ DSTAGNN forward pass successful")
        print(f"  Input shape: {batch['X'].shape}")
        print(f"  Output shape: {output.shape}")
        return True

    except Exception as e:
        print(f"✗ DSTAGNN forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dstagnn_loss():
    """Test if DSTAGNN loss calculation works"""
    try:
        from libcity.model.traffic_flow_prediction import DSTAGNN

        # Setup (same as previous tests)
        class DummyConfig:
            def get(self, key, default=None):
                config_dict = {
                    'device': torch.device('cpu'),
                    'input_window': 12,
                    'output_window': 12,
                    'in_channels': 1,
                    'nb_block': 2,
                    'K': 3,
                    'nb_chev_filter': 16,
                    'nb_time_filter': 16,
                    'd_model': 64,
                    'd_k': 16,
                    'n_heads': 2,
                    'graph_use': 'AG',
                }
                return config_dict.get(key, default)

        num_nodes = 10
        adj_mx = np.random.rand(num_nodes, num_nodes)
        adj_mx = (adj_mx + adj_mx.T) / 2

        class DummyScaler:
            def inverse_transform(self, data):
                return data

        class DummyDataFeature:
            def get(self, key, default=None):
                feature_dict = {
                    'num_nodes': num_nodes,
                    'adj_mx': adj_mx,
                    'adj_TMD': adj_mx,
                    'adj_pa': adj_mx,
                    'scaler': DummyScaler(),
                }
                return feature_dict.get(key, default)

        config = DummyConfig()
        data_feature = DummyDataFeature()
        model = DSTAGNN(config, data_feature)

        # Create dummy batch
        batch_size = 4
        input_window = 12
        output_window = 12

        batch = {
            'X': torch.randn(batch_size, input_window, num_nodes, 1),
            'y': torch.randn(batch_size, output_window, num_nodes, 1),
        }

        # Test loss calculation
        loss = model.calculate_loss(batch)

        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"

        print(f"✓ DSTAGNN loss calculation successful")
        print(f"  Loss value: {loss.item():.4f}")
        return True

    except Exception as e:
        print(f"✗ DSTAGNN loss calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("DSTAGNN Model Adaptation Tests")
    print("=" * 60)

    results = []

    print("\n1. Testing Import...")
    results.append(test_dstagnn_import())

    print("\n2. Testing Initialization...")
    results.append(test_dstagnn_initialization())

    print("\n3. Testing Forward Pass...")
    results.append(test_dstagnn_forward())

    print("\n4. Testing Loss Calculation...")
    results.append(test_dstagnn_loss())

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("\n✓ All tests passed! DSTAGNN is ready for use.")
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
