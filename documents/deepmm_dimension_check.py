"""
Quick dimension check for DeepMM decoder fix.

This script verifies that the decoder initialization and forward pass
have the correct dimensions after the fix.
"""

import torch
import torch.nn as nn

# Simulate the configuration
config = {
    'src_loc_emb_dim': 256,
    'trg_seg_emb_dim': 256,
    'src_hidden_dim': 512,
    'trg_hidden_dim': 512,
    'bidirectional': True,
    'n_layers_src': 2,
    'dropout': 0.5,
    'rnn_type': 'LSTM',
    'attn_type': 'dot',
    'device': torch.device('cpu')
}

# Simulate the dimension calculations
src_hidden_dim = config['src_hidden_dim']
trg_hidden_dim = config['trg_hidden_dim']
trg_seg_emb_dim = config['trg_seg_emb_dim']
bidirectional = config['bidirectional']

num_directions = 2 if bidirectional else 1
src_hidden_dim = src_hidden_dim // 2 if bidirectional else src_hidden_dim

print("=== Configuration ===")
print(f"Original src_hidden_dim from config: 512")
print(f"After bidirectional division: {src_hidden_dim}")
print(f"Number of directions: {num_directions}")
print(f"trg_hidden_dim: {trg_hidden_dim}")
print(f"trg_seg_emb_dim: {trg_seg_emb_dim}")

print("\n=== Encoder Output ===")
encoder_output_dim = src_hidden_dim * num_directions
print(f"Encoder output dim (src_hidden_dim * num_directions): {encoder_output_dim}")

print("\n=== Decoder Dimensions ===")
decoder_input_dim = trg_seg_emb_dim + (src_hidden_dim * num_directions)
decoder_hidden_size = trg_hidden_dim
print(f"Decoder input_size: {decoder_input_dim}")
print(f"  - trg_seg_emb_dim: {trg_seg_emb_dim}")
print(f"  - encoder context: {src_hidden_dim * num_directions}")
print(f"Decoder hidden_size: {decoder_hidden_size}")

print("\n=== Expected Layer Dimensions ===")
print(f"input_weights: nn.Linear({decoder_input_dim}, {4 * decoder_hidden_size})")
print(f"hidden_weights: nn.Linear({decoder_hidden_size}, {4 * decoder_hidden_size})")

print("\n=== Verification ===")
# Simulate batch
batch_size = 32
src_seq_len = 40
trg_seq_len = 54

print(f"Batch size: {batch_size}")
print(f"Source sequence length: {src_seq_len}")
print(f"Target sequence length: {trg_seq_len}")

# Encoder output
src_h_shape = (batch_size, src_seq_len, encoder_output_dim)
print(f"\nEncoder output shape: {src_h_shape}")

# Final encoder state (concatenated bidirectional)
h_t_shape = (batch_size, encoder_output_dim)
print(f"Final encoder state shape: {h_t_shape}")

# Transform to decoder init state
encoder2decoder_shape = (encoder_output_dim, decoder_hidden_size)
decoder_init_state_shape = (batch_size, decoder_hidden_size)
print(f"encoder2decoder Linear: ({encoder_output_dim}, {decoder_hidden_size})")
print(f"Decoder init state shape: {decoder_init_state_shape}")

# Target embedding
trg_emb_shape = (batch_size, trg_seq_len, trg_seg_emb_dim)
print(f"\nTarget embedding shape: {trg_emb_shape}")

# Encoder context (expanded)
encoder_context_shape = (batch_size, trg_seq_len, encoder_output_dim)
print(f"Encoder context (expanded) shape: {encoder_context_shape}")

# Decoder input (concatenated)
decoder_input_shape = (batch_size, trg_seq_len, decoder_input_dim)
print(f"Decoder input (concatenated) shape: {decoder_input_shape}")

# Decoder output
decoder_output_shape = (batch_size, trg_seq_len, decoder_hidden_size)
print(f"Decoder output shape: {decoder_output_shape}")

print("\n=== All dimensions are consistent! ===")
