"""
RNTrajRec: Road Network-aware Trajectory Recovery

This model is adapted from the original RNTrajRec implementation for trajectory
recovery/map matching tasks. The original model uses a transformer encoder with
road network graph refinement and an attention-based GRU decoder.

Original paper: "RNTrajRec: Road Network Enhanced Trajectory Recovery
with Spatial-Temporal Transformer"

Original repository: https://github.com/WenMellors/RNTrajRec

Key Components:
1. PositionalEncoder - Sinusoidal positional encoding for sequences
2. MultiHeadAttention - Multi-head self-attention mechanism
3. TransformerEncoderLayer - Transformer encoder layer with FFN
4. TransformerEncoder - Stack of encoder layers with positional encoding
5. Attention - Bahdanau-style attention for decoder
6. TrajDecoder - GRU-based decoder with attention mechanism
7. RNTrajRec - Main seq2seq model for trajectory recovery

Adaptations for LibCity:
- Removed DGL dependency for graph neural networks
- Simplified road network handling to work with LibCity's location vocabulary
- Replaced RoadGNN with learnable location embeddings
- Removed graph refinement layers (requires DGL batched graphs)
- Adapted batch input format to LibCity's trajectory batch dictionary
- Implemented predict() and calculate_loss() methods following LibCity conventions
- Added teacher forcing ratio control for training

Limitations compared to original:
- No explicit road network graph structure (simplified to embedding-based)
- No graph refinement between transformer layers
- No constraint mask from road network connectivity
- Simplified rate prediction (original predicts sub-segment position)

The adapted model can still perform trajectory location prediction but without
the road network topology constraints of the original model.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from libcity.model.abstract_model import AbstractModel


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for transformer.

    Args:
        d_model: Model dimension
        device: Computation device
        max_seq_len: Maximum sequence length
    """

    def __init__(self, d_model, device, max_seq_len=500):
        super().__init__()
        self.d_model = d_model
        self.device = device

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Apply positional encoding.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).to(self.device)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Args:
        heads: Number of attention heads
        d_model: Model dimension
        dropout: Dropout probability
    """

    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """Forward pass.

        Args:
            q: Query tensor (batch, seq_len, d_model)
            k: Key tensor (batch, seq_len, d_model)
            v: Value tensor (batch, seq_len, d_model)
            mask: Attention mask (batch, seq_len, seq_len)

        Returns:
            Attended output (batch, seq_len, d_model)
        """
        bs = q.size(0)

        # Linear projections and split into heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # Transpose for attention: (bs, heads, seq_len, d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        output = torch.matmul(scores, v)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        return self.out(output)


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class LayerNorm(nn.Module):
    """Layer normalization.

    Args:
        d_model: Model dimension
        eps: Epsilon for numerical stability
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer.

    Args:
        d_model: Model dimension
        heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model, heads=8, d_ff=512, dropout=0.1):
        super().__init__()
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Attention mask (batch, seq_len, seq_len)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder with positional encoding.

    Args:
        d_model: Model dimension
        num_layers: Number of encoder layers
        device: Computation device
        heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
        max_seq_len: Maximum sequence length
    """

    def __init__(self, d_model, num_layers, device, heads=8, d_ff=512,
                 dropout=0.1, max_seq_len=500):
        super().__init__()
        self.num_layers = num_layers
        self.pe = PositionalEncoder(d_model, device, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)

    def forward(self, src, mask=None):
        """Forward pass.

        Args:
            src: Source sequence (batch, seq_len, d_model)
            mask: Attention mask (batch, seq_len, seq_len)

        Returns:
            Encoded output (batch, seq_len, d_model)
        """
        x = self.pe(src)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderAttention(nn.Module):
    """Bahdanau-style attention for decoder.

    Calculates attention between decoder hidden state and encoder outputs.

    Args:
        hid_dim: Hidden dimension
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, attn_mask=None):
        """Calculate attention weights.

        Args:
            hidden: Decoder hidden state (1, batch, hid_dim)
            encoder_outputs: Encoder outputs (seq_len, batch, hid_dim)
            attn_mask: Attention mask (batch, seq_len)

        Returns:
            Attention weights (batch, seq_len)
        """
        src_len = encoder_outputs.shape[0]

        # Repeat hidden state for each source position
        hidden = hidden.repeat(src_len, 1, 1)
        hidden = hidden.permute(1, 0, 2)  # (batch, seq_len, hid_dim)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # (batch, seq_len, hid_dim)

        # Calculate attention energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # (batch, seq_len)

        # Apply mask
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask == 0, -1e6)

        return F.softmax(attention, dim=1)


class TrajDecoder(nn.Module):
    """Trajectory decoder with attention.

    GRU-based decoder that uses attention over encoder outputs to predict
    the next location in the trajectory.

    Args:
        loc_size: Number of locations in vocabulary
        loc_emb_dim: Location embedding dimension
        hid_dim: Hidden dimension
        dropout: Dropout probability
        use_attention: Whether to use attention mechanism
    """

    def __init__(self, loc_size, loc_emb_dim, hid_dim, dropout=0.1, use_attention=True):
        super().__init__()
        self.loc_size = loc_size
        self.loc_emb_dim = loc_emb_dim
        self.hid_dim = hid_dim
        self.use_attention = use_attention

        # Location embedding
        self.emb_loc = nn.Embedding(loc_size, loc_emb_dim, padding_idx=0)

        # RNN input dimension
        rnn_input_dim = loc_emb_dim
        if use_attention:
            self.attn = DecoderAttention(hid_dim)
            rnn_input_dim += hid_dim

        self.rnn = nn.GRU(rnn_input_dim, hid_dim, batch_first=False)
        self.fc_out = nn.Linear(hid_dim, loc_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_loc, hidden, encoder_outputs, attn_mask=None):
        """Decoder forward step.

        Args:
            input_loc: Input location indices (batch, 1)
            hidden: Decoder hidden state (1, batch, hid_dim)
            encoder_outputs: Encoder outputs (seq_len, batch, hid_dim)
            attn_mask: Attention mask (batch, seq_len)

        Returns:
            prediction: Location prediction logits (batch, loc_size)
            hidden: Updated hidden state (1, batch, hid_dim)
        """
        input_loc = input_loc.squeeze(1)  # (batch,)

        # Embed input location
        embedded = self.dropout(self.emb_loc(input_loc)).unsqueeze(0)  # (1, batch, emb_dim)

        if self.use_attention:
            # Calculate attention
            a = self.attn(hidden, encoder_outputs, attn_mask)  # (batch, seq_len)
            a = a.unsqueeze(1)  # (batch, 1, seq_len)

            encoder_outputs_perm = encoder_outputs.permute(1, 0, 2)  # (batch, seq_len, hid_dim)
            weighted = torch.bmm(a, encoder_outputs_perm)  # (batch, 1, hid_dim)
            weighted = weighted.permute(1, 0, 2)  # (1, batch, hid_dim)

            rnn_input = torch.cat((embedded, weighted), dim=2)
        else:
            rnn_input = embedded

        output, hidden = self.rnn(rnn_input, hidden)

        # Predict next location
        prediction = self.fc_out(output.squeeze(0))  # (batch, loc_size)

        return prediction, hidden


class TrajEncoder(nn.Module):
    """Trajectory encoder with optional temporal and spatial features.

    Combines location embeddings with optional time embeddings and processes
    through a transformer encoder.

    Args:
        loc_size: Number of locations
        loc_emb_dim: Location embedding dimension
        hid_dim: Hidden dimension
        num_layers: Number of transformer layers
        device: Computation device
        use_time: Whether to use time embeddings
        tim_size: Number of time slots (if use_time)
        tim_emb_dim: Time embedding dimension (if use_time)
        heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, loc_size, loc_emb_dim, hid_dim, num_layers, device,
                 use_time=True, tim_size=48, tim_emb_dim=64, heads=8, dropout=0.1):
        super().__init__()
        self.hid_dim = hid_dim
        self.device = device
        self.use_time = use_time

        # Location embedding
        self.emb_loc = nn.Embedding(loc_size, loc_emb_dim, padding_idx=0)

        # Time embedding (optional)
        if use_time:
            self.emb_tim = nn.Embedding(tim_size, tim_emb_dim)
            input_dim = loc_emb_dim + tim_emb_dim
        else:
            input_dim = loc_emb_dim

        # Input projection
        self.fc_in = nn.Linear(input_dim, hid_dim)

        # Transformer encoder
        self.transformer = TransformerEncoder(
            d_model=hid_dim,
            num_layers=num_layers,
            device=device,
            heads=heads,
            d_ff=hid_dim * 2,
            dropout=dropout
        )

        # Hidden state projection
        self.fc_hid = nn.Linear(hid_dim, hid_dim)

    def forward(self, loc_seq, tim_seq=None, src_len=None):
        """Encode trajectory sequence.

        Args:
            loc_seq: Location sequence (batch, seq_len)
            tim_seq: Time sequence (batch, seq_len), optional
            src_len: Original sequence lengths (batch,)

        Returns:
            outputs: Encoder outputs (seq_len, batch, hid_dim)
            hidden: Final hidden state (1, batch, hid_dim)
        """
        batch_size = loc_seq.size(0)
        max_src_len = loc_seq.size(1)

        # Embed locations
        loc_emb = self.emb_loc(loc_seq)  # (batch, seq_len, loc_emb_dim)

        # Combine with time embeddings if available
        if self.use_time and tim_seq is not None:
            tim_emb = self.emb_tim(tim_seq)  # (batch, seq_len, tim_emb_dim)
            src = torch.cat([loc_emb, tim_emb], dim=-1)
        else:
            src = loc_emb

        # Project to hidden dimension
        src = self.fc_in(src)  # (batch, seq_len, hid_dim)

        # Create attention mask
        if src_len is not None:
            mask = torch.zeros(batch_size, max_src_len, max_src_len).to(self.device)
            for i in range(batch_size):
                mask[i, :src_len[i], :src_len[i]] = 1
        else:
            mask = None

        # Encode with transformer
        outputs = self.transformer(src, mask)  # (batch, seq_len, hid_dim)

        # Compute mean hidden state (excluding padding)
        if src_len is not None:
            hidden_list = []
            for i in range(batch_size):
                valid_outputs = outputs[i, :src_len[i], :]
                hidden_list.append(valid_outputs.mean(dim=0))
            hidden = torch.stack(hidden_list, dim=0).unsqueeze(0)  # (1, batch, hid_dim)
        else:
            hidden = outputs.mean(dim=1).unsqueeze(0)  # (1, batch, hid_dim)

        hidden = torch.tanh(self.fc_hid(hidden))

        # Transpose to (seq_len, batch, hid_dim) for decoder compatibility
        outputs = outputs.permute(1, 0, 2)

        return outputs, hidden


class RNTrajRec(AbstractModel):
    """
    RNTrajRec: Road Network-aware Trajectory Recovery Model

    This is a simplified adaptation of the original RNTrajRec model for LibCity.
    The model uses a transformer encoder to process input trajectories and an
    attention-based GRU decoder to predict recovered trajectory locations.

    The original model includes road network graph neural networks for spatial
    reasoning, which has been simplified to learnable embeddings in this adaptation.

    Args:
        config (dict): Configuration dictionary containing model hyperparameters
        data_feature (dict): Data features including vocabulary sizes

    Required config parameters:
        - hid_dim: Hidden dimension (default: 512)
        - loc_emb_dim: Location embedding dimension (default: 512)
        - transformer_layers: Number of transformer encoder layers (default: 2)
        - num_heads: Number of attention heads (default: 8)
        - dropout: Dropout probability (default: 0.1)
        - use_attention: Whether to use attention in decoder (default: True)
        - use_time: Whether to use time embeddings (default: True)
        - tim_emb_dim: Time embedding dimension (default: 64)
        - teacher_forcing_ratio: Teacher forcing ratio for training (default: 0.5)
        - max_output_len: Maximum output sequence length (default: 128)

    Required data_feature:
        - loc_size: Number of location tokens in vocabulary
        - tim_size: Number of time slots (if use_time)
        - loc_pad: Padding index for locations
    """

    def __init__(self, config, data_feature):
        super(RNTrajRec, self).__init__(config, data_feature)

        self.device = config.get('device', 'cpu')

        # Data dimensions from data_feature
        self.loc_size = data_feature.get('loc_size', 1000)
        self.tim_size = data_feature.get('tim_size', 48)
        self.loc_pad = data_feature.get('loc_pad', 0)

        # Model hyperparameters from config
        self.hid_dim = config.get('hid_dim', 512)
        self.loc_emb_dim = config.get('loc_emb_dim', 512)
        self.transformer_layers = config.get('transformer_layers', 2)
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.1)
        self.use_attention = config.get('use_attention', True)
        self.use_time = config.get('use_time', True)
        self.tim_emb_dim = config.get('tim_emb_dim', 64)
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0.5)
        self.max_output_len = config.get('max_output_len', 128)

        # Evaluation method
        self.evaluate_method = config.get('evaluate_method', 'all')

        # Build model components
        self._build_model()

    def _build_model(self):
        """Build encoder and decoder components."""

        # Trajectory Encoder
        self.encoder = TrajEncoder(
            loc_size=self.loc_size,
            loc_emb_dim=self.loc_emb_dim,
            hid_dim=self.hid_dim,
            num_layers=self.transformer_layers,
            device=self.device,
            use_time=self.use_time,
            tim_size=self.tim_size,
            tim_emb_dim=self.tim_emb_dim,
            heads=self.num_heads,
            dropout=self.dropout
        )

        # Trajectory Decoder
        self.decoder = TrajDecoder(
            loc_size=self.loc_size,
            loc_emb_dim=self.loc_emb_dim,
            hid_dim=self.hid_dim,
            dropout=self.dropout,
            use_attention=self.use_attention
        )

    def forward(self, batch, teacher_forcing_ratio=None):
        """
        Forward pass for trajectory recovery.

        Args:
            batch: LibCity Batch object containing trajectory data. Expected keys:
                - 'current_loc': Input location sequence (batch, src_len)
                - 'current_tim': Input time sequence (batch, src_len), optional
                - 'target': Target location for next-step prediction (batch,)
                  OR 'target_loc': Full target sequence (batch, trg_len) for recovery

        Returns:
            outputs: Location prediction logits (batch, trg_len, loc_size)
        """
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.teacher_forcing_ratio if self.training else 0.0

        # Extract data from batch
        src_loc = batch['current_loc']
        if 'current_tim' in batch.data:
            src_tim = batch['current_tim']
        else:
            src_tim = None

        # Get sequence lengths
        if hasattr(batch, 'get_origin_len'):
            src_len = batch.get_origin_len('current_loc')
            if isinstance(src_len, torch.Tensor):
                src_len = src_len.tolist()
        else:
            src_len = None

        # Convert to tensors if needed
        if not isinstance(src_loc, torch.Tensor):
            src_loc = torch.LongTensor(src_loc)
        src_loc = src_loc.to(self.device)

        if src_tim is not None and not isinstance(src_tim, torch.Tensor):
            src_tim = torch.LongTensor(src_tim)
        if src_tim is not None:
            src_tim = src_tim.to(self.device)

        batch_size = src_loc.size(0)
        max_src_len = src_loc.size(1)

        # Encode input trajectory
        encoder_outputs, hidden = self.encoder(src_loc, src_tim, src_len)

        # Prepare attention mask for decoder
        if src_len is not None:
            attn_mask = torch.zeros(batch_size, max_src_len).to(self.device)
            for i in range(batch_size):
                attn_mask[i, :src_len[i]] = 1
        else:
            attn_mask = torch.ones(batch_size, max_src_len).to(self.device)

        # Determine target sequence for decoding
        if 'target_loc' in batch.data:
            # Full sequence recovery mode
            trg_loc = batch['target_loc']
            if not isinstance(trg_loc, torch.Tensor):
                trg_loc = torch.LongTensor(trg_loc)
            trg_loc = trg_loc.to(self.device)
            max_trg_len = trg_loc.size(1)
        else:
            # Next-step prediction mode
            trg_loc = None
            max_trg_len = 2  # Just predict one step

        # Initialize decoder outputs
        outputs = torch.zeros(batch_size, max_trg_len, self.loc_size).to(self.device)

        # First decoder input: last location from source or special token
        if src_len is not None:
            input_loc = torch.zeros(batch_size, 1, dtype=torch.long).to(self.device)
            for i in range(batch_size):
                input_loc[i, 0] = src_loc[i, src_len[i] - 1]
        else:
            input_loc = src_loc[:, -1:].clone()

        # Decode step by step
        for t in range(1, max_trg_len):
            prediction, hidden = self.decoder(
                input_loc, hidden, encoder_outputs, attn_mask
            )
            outputs[:, t, :] = prediction

            # Teacher forcing decision
            teacher_force = random.random() < teacher_forcing_ratio

            # Get next input
            if teacher_force and trg_loc is not None:
                input_loc = trg_loc[:, t:t+1]
            else:
                input_loc = prediction.argmax(dim=1).unsqueeze(1)

        return outputs

    def predict(self, batch):
        """
        Prediction method for LibCity evaluation.

        Args:
            batch: Input batch dictionary

        Returns:
            POI prediction scores (batch, loc_size) for next location
        """
        # Forward pass without teacher forcing
        outputs = self.forward(batch, teacher_forcing_ratio=0.0)

        # Get prediction for last timestep (index 1 in outputs since index 0 is empty)
        if outputs.size(1) > 1:
            scores = outputs[:, 1, :]  # (batch, loc_size)
        else:
            scores = outputs[:, 0, :]

        # Apply log softmax
        scores = F.log_softmax(scores, dim=-1)

        if self.evaluate_method == 'sample':
            # Handle negative sampling evaluation
            if 'neg_loc' in batch.data:
                pos_neg_index = torch.cat(
                    (batch['target'].unsqueeze(1), batch['neg_loc']), dim=1
                )
                scores = torch.gather(scores, 1, pos_neg_index)

        return scores

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Args:
            batch: LibCity Batch object containing:
                - trajectory data for forward pass
                - 'target': Target location indices (batch,)

        Returns:
            Cross-entropy loss for next location prediction
        """
        # Forward pass with teacher forcing
        outputs = self.forward(batch)

        # Get target
        if 'target' in batch.data:
            target = batch['target']
            if not isinstance(target, torch.Tensor):
                target = torch.LongTensor(target)
            target = target.to(self.device)

            # Get prediction at last step
            if outputs.size(1) > 1:
                pred = outputs[:, 1, :]  # (batch, loc_size)
            else:
                pred = outputs[:, 0, :]

            # Cross-entropy loss
            criterion = nn.CrossEntropyLoss(ignore_index=self.loc_pad)
            loss = criterion(pred, target)

        elif 'target_loc' in batch.data:
            # Full sequence recovery mode
            trg_loc = batch['target_loc']
            if not isinstance(trg_loc, torch.Tensor):
                trg_loc = torch.LongTensor(trg_loc)
            trg_loc = trg_loc.to(self.device)

            # Get target lengths if available
            if hasattr(batch, 'get_origin_len'):
                trg_len = batch.get_origin_len('target_loc')
                if isinstance(trg_len, torch.Tensor):
                    trg_len = trg_len.tolist()
            else:
                trg_len = [trg_loc.size(1)] * trg_loc.size(0)

            # Calculate loss for each position
            criterion = nn.CrossEntropyLoss(ignore_index=self.loc_pad)

            # Flatten predictions and targets
            pred_flat = outputs[:, 1:, :].contiguous().view(-1, self.loc_size)
            target_flat = trg_loc[:, 1:].contiguous().view(-1)

            loss = criterion(pred_flat, target_flat)

        else:
            raise ValueError("Batch must contain either 'target' or 'target_loc'")

        return loss

    def recover_trajectory(self, batch, max_len=None):
        """
        Recover full trajectory from sparse observations.

        This is the main trajectory recovery function that generates a complete
        trajectory from sparse GPS observations.

        Args:
            batch: Input batch with sparse trajectory observations
            max_len: Maximum output length (default: self.max_output_len)

        Returns:
            recovered_locs: Recovered location indices (batch, max_len)
            probs: Prediction probabilities (batch, max_len, loc_size)
        """
        if max_len is None:
            max_len = self.max_output_len

        self.eval()
        with torch.no_grad():
            # Extract data
            src_loc = batch['current_loc']
            if 'current_tim' in batch.data:
                src_tim = batch['current_tim']
            else:
                src_tim = None

            if hasattr(batch, 'get_origin_len'):
                src_len = batch.get_origin_len('current_loc')
                if isinstance(src_len, torch.Tensor):
                    src_len = src_len.tolist()
            else:
                src_len = None

            if not isinstance(src_loc, torch.Tensor):
                src_loc = torch.LongTensor(src_loc)
            src_loc = src_loc.to(self.device)

            if src_tim is not None:
                if not isinstance(src_tim, torch.Tensor):
                    src_tim = torch.LongTensor(src_tim)
                src_tim = src_tim.to(self.device)

            batch_size = src_loc.size(0)
            max_src_len = src_loc.size(1)

            # Encode
            encoder_outputs, hidden = self.encoder(src_loc, src_tim, src_len)

            # Attention mask
            if src_len is not None:
                attn_mask = torch.zeros(batch_size, max_src_len).to(self.device)
                for i in range(batch_size):
                    attn_mask[i, :src_len[i]] = 1
            else:
                attn_mask = torch.ones(batch_size, max_src_len).to(self.device)

            # Initialize outputs
            recovered_locs = torch.zeros(batch_size, max_len, dtype=torch.long).to(self.device)
            probs = torch.zeros(batch_size, max_len, self.loc_size).to(self.device)

            # Initial input
            if src_len is not None:
                input_loc = torch.zeros(batch_size, 1, dtype=torch.long).to(self.device)
                for i in range(batch_size):
                    input_loc[i, 0] = src_loc[i, src_len[i] - 1]
                    recovered_locs[i, 0] = src_loc[i, 0]  # Start token
            else:
                input_loc = src_loc[:, -1:].clone()
                recovered_locs[:, 0] = src_loc[:, 0]

            # Decode autoregressively
            for t in range(1, max_len):
                prediction, hidden = self.decoder(
                    input_loc, hidden, encoder_outputs, attn_mask
                )

                probs[:, t, :] = F.softmax(prediction, dim=-1)
                top1 = prediction.argmax(dim=1)
                recovered_locs[:, t] = top1
                input_loc = top1.unsqueeze(1)

        return recovered_locs, probs
