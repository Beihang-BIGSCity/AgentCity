"""
JCLRNT: Joint Contrastive Learning for Road Network and Trajectory Representation

This module implements the JCLRNT model for joint representation learning of
road networks and trajectories. The model uses:
1. GraphEncoder (GATConv) for learning node/road segment representations
2. TransformerModel for learning trajectory representations
3. Three contrastive losses: node-to-node, trajectory-to-trajectory, node-to-trajectory

Original paper: JCLRNT - Joint Contrastive Learning for Road Network and Trajectory

Adapted from: repos/JCLRNT/models/sv.py

Key Components:
    - GraphEncoder: GATConv-based graph encoder with 2 layers
    - TransformerModel: Transformer encoder with positional encoding
    - Three contrastive loss functions: JSD, NCE, NTX

Data Format:
    Input batch is a dictionary containing (LibCity TrajectoryDataset format):
        - 'current_loc': Current location sequences (batch_size, seq_len) with location IDs
        - 'current_tim': Current time sequences (batch_size, seq_len)
        - 'history_loc': Historical location sequences
        - 'history_tim': Historical time sequences
        - 'target': Target location IDs
        - 'uid': User IDs
    For backward compatibility, 'X' key is also supported:
        - 'X': Trajectory sequences (batch_size, max_seq_len) with road segment indices
    Graph structure (from data_feature):
        - 'edge_index': Graph edge indices (2, num_edges)

Configuration Parameters:
    - embed_size: Embedding dimension (default: 128)
    - hidden_size: Hidden layer dimension (default: 128)
    - drop_rate: Dropout rate for transformer (default: 0.2)
    - drop_edge_rate: Edge dropout rate for graph augmentation (default: 0.2)
    - drop_road_rate: Road masking rate for sequence augmentation (default: 0.2)
    - lambda_st: Weight for spatial-temporal (node-trajectory) loss (default: 0.8)
    - loss_measure: Contrastive loss type - "jsd", "nce", or "ntx" (default: "jsd")
    - mode: Encoding mode - "s" (structural with GNN) or "p" (pure embeddings) (default: "s")
    - num_heads: Number of attention heads in transformer (default: 4)
    - num_transformer_layers: Number of transformer encoder layers (default: 2)
    - num_graph_layers: Number of GNN layers (default: 2)
    - weighted_loss: Whether to use weighted node-seq loss (default: False)
    - activation: Activation function - "relu" or "prelu" (default: "relu")

Authors: Adapted from JCLRNT repository for LibCity
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger

try:
    from torch_geometric.nn import GATConv
    from torch_geometric.utils import dropout_adj
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False
    GATConv = None

from libcity.model.abstract_model import AbstractModel


def jsd(z1, z2, pos_mask):
    """Jensen-Shannon Divergence based contrastive loss.

    Args:
        z1: First view embeddings (N, D)
        z2: Second view embeddings (M, D)
        pos_mask: Positive pair mask (N, M)

    Returns:
        JSD loss scalar
    """
    neg_mask = 1 - pos_mask
    sim_mat = torch.mm(z1, z2.t())
    E_pos = math.log(2.) - F.softplus(-sim_mat)
    E_neg = F.softplus(-sim_mat) + sim_mat - math.log(2.)
    neg_sum = neg_mask.sum()
    pos_sum = pos_mask.sum()
    if neg_sum > 0 and pos_sum > 0:
        return (E_neg * neg_mask).sum() / neg_sum - (E_pos * pos_mask).sum() / pos_sum
    return torch.tensor(0.0, device=z1.device)


def nce(z1, z2, pos_mask):
    """Noise Contrastive Estimation loss.

    Args:
        z1: First view embeddings (N, D)
        z2: Second view embeddings (M, D)
        pos_mask: Positive pair mask (N, M)

    Returns:
        NCE loss scalar
    """
    sim_mat = torch.mm(z1, z2.t())
    return nn.BCEWithLogitsLoss(reduction='none')(sim_mat, pos_mask).sum(1).mean()


def ntx(z1, z2, pos_mask, tau=0.5, normalize=False):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    Args:
        z1: First view embeddings (N, D)
        z2: Second view embeddings (M, D)
        pos_mask: Positive pair mask (N, M)
        tau: Temperature parameter
        normalize: Whether to L2 normalize embeddings

    Returns:
        NT-Xent loss scalar
    """
    if normalize:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    sim_mat = torch.mm(z1, z2.t())
    sim_mat = torch.exp(sim_mat / tau)
    pos_sum = pos_mask.sum(1)
    pos_sum = torch.clamp(pos_sum, min=1e-9)
    return -torch.log((sim_mat * pos_mask).sum(1) / sim_mat.sum(1) / pos_sum).mean()


def node_node_loss(node_rep1, node_rep2, measure, device):
    """Compute node-to-node contrastive loss.

    Positive pairs are the same node in two augmented views.

    Args:
        node_rep1: Node representations from view 1 (num_nodes, D)
        node_rep2: Node representations from view 2 (num_nodes, D)
        measure: Loss measure type ("jsd", "nce", "ntx")
        device: Device for tensors

    Returns:
        Node-to-node contrastive loss
    """
    num_nodes = node_rep1.shape[0]
    pos_mask = torch.eye(num_nodes, device=device)

    if measure == 'jsd':
        return jsd(node_rep1, node_rep2, pos_mask)
    elif measure == 'nce':
        return nce(node_rep1, node_rep2, pos_mask)
    elif measure == 'ntx':
        return ntx(node_rep1, node_rep2, pos_mask)
    return jsd(node_rep1, node_rep2, pos_mask)


def seq_seq_loss(seq_rep1, seq_rep2, measure, device):
    """Compute sequence-to-sequence (trajectory) contrastive loss.

    Positive pairs are the same trajectory in two augmented views.

    Args:
        seq_rep1: Trajectory representations from view 1 (batch_size, D)
        seq_rep2: Trajectory representations from view 2 (batch_size, D)
        measure: Loss measure type ("jsd", "nce", "ntx")
        device: Device for tensors

    Returns:
        Trajectory-to-trajectory contrastive loss
    """
    batch_size = seq_rep1.shape[0]
    pos_mask = torch.eye(batch_size, device=device)

    if measure == 'jsd':
        return jsd(seq_rep1, seq_rep2, pos_mask)
    elif measure == 'nce':
        return nce(seq_rep1, seq_rep2, pos_mask)
    elif measure == 'ntx':
        return ntx(seq_rep1, seq_rep2, pos_mask)
    return jsd(seq_rep1, seq_rep2, pos_mask)


def node_seq_loss(node_rep, seq_rep, sequences, measure, device, padding_idx):
    """Compute node-to-sequence contrastive loss.

    Positive pairs are nodes that appear in the trajectory.

    Args:
        node_rep: Node representations (num_nodes, D)
        seq_rep: Trajectory representations (batch_size, D)
        sequences: Trajectory sequences (batch_size, max_seq_len)
        measure: Loss measure type ("jsd", "nce", "ntx")
        device: Device for tensors
        padding_idx: Padding index to ignore

    Returns:
        Node-to-sequence contrastive loss
    """
    batch_size = seq_rep.shape[0]
    num_nodes = node_rep.shape[0]

    pos_mask = torch.zeros((batch_size, num_nodes), device=device)
    for row_idx, row in enumerate(sequences):
        valid_nodes = row[row != padding_idx]
        if len(valid_nodes) > 0:
            valid_nodes = valid_nodes[valid_nodes < num_nodes]
            pos_mask[row_idx, valid_nodes] = 1.0

    if measure == 'jsd':
        return jsd(seq_rep, node_rep, pos_mask)
    elif measure == 'nce':
        return nce(seq_rep, node_rep, pos_mask)
    elif measure == 'ntx':
        return ntx(seq_rep, node_rep, pos_mask)
    return jsd(seq_rep, node_rep, pos_mask)


def weighted_ns_loss(node_rep, seq_rep, weights, measure):
    """Compute weighted node-to-sequence contrastive loss.

    Args:
        node_rep: Node representations (num_nodes, D)
        seq_rep: Trajectory representations (batch_size, D)
        weights: Weight matrix for positive pairs (batch_size, num_nodes)
        measure: Loss measure type ("jsd", "nce", "ntx")

    Returns:
        Weighted node-to-sequence contrastive loss
    """
    if measure == 'jsd':
        return jsd(seq_rep, node_rep, weights)
    elif measure == 'nce':
        return nce(seq_rep, node_rep, weights)
    elif measure == 'ntx':
        return ntx(seq_rep, node_rep, weights)
    return jsd(seq_rep, node_rep, weights)


def random_mask(x, mask_token, mask_prob=0.2):
    """Apply random masking to sequence for data augmentation.

    Args:
        x: Input sequence tensor
        mask_token: Token to use for masking (usually padding_idx)
        mask_prob: Probability of masking each position

    Returns:
        Masked sequence tensor
    """
    mask_pos = torch.empty(
        x.size(),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < mask_prob
    x = x.clone()
    x[mask_pos] = mask_token
    return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer.

    Args:
        dim: Embedding dimension
        dropout: Dropout probability
        max_len: Maximum sequence length
    """

    def __init__(self, dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input.

        Args:
            x: Input tensor (seq_len, batch_size, dim)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer encoder for trajectory representation.

    Args:
        input_size: Input feature dimension
        num_heads: Number of attention heads
        hidden_size: Hidden dimension in feed-forward layers
        num_layers: Number of transformer encoder layers
        dropout: Dropout probability
    """

    def __init__(self, input_size, num_heads, hidden_size, num_layers, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src, src_mask, src_key_padding_mask):
        """Forward pass through transformer encoder.

        Args:
            src: Source sequence (seq_len, batch_size, dim)
            src_mask: Attention mask (unused, can be None)
            src_key_padding_mask: Padding mask (batch_size, seq_len)

        Returns:
            Encoded sequence (seq_len, batch_size, dim)
        """
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        return output


class GraphEncoder(nn.Module):
    """Graph encoder using GATConv layers.

    Args:
        input_size: Input feature dimension
        output_size: Output feature dimension
        encoder_layer: GNN layer class (GATConv)
        num_layers: Number of GNN layers
        activation: Activation function
    """

    def __init__(self, input_size, output_size, encoder_layer, num_layers, activation):
        super(GraphEncoder, self).__init__()

        self.num_layers = num_layers
        self.activation = activation

        if encoder_layer is not None:
            self.layers = nn.ModuleList([encoder_layer(input_size, output_size)])
            for _ in range(1, num_layers):
                self.layers.append(encoder_layer(output_size, output_size))
        else:
            # Fallback to linear layers if PyG not available
            self.layers = nn.ModuleList([nn.Linear(input_size, output_size)])
            for _ in range(1, num_layers):
                self.layers.append(nn.Linear(output_size, output_size))
            self._use_linear = True

    def forward(self, x, edge_index):
        """Forward pass through graph encoder.

        Args:
            x: Node features (num_nodes, input_size)
            edge_index: Edge indices (2, num_edges)

        Returns:
            Encoded node features (num_nodes, output_size)
        """
        for i in range(self.num_layers):
            if hasattr(self, '_use_linear') and self._use_linear:
                x = self.activation(self.layers[i](x))
            else:
                x = self.activation(self.layers[i](x, edge_index))
        return x


class SingleViewModel(nn.Module):
    """Single-view contrastive learning model for road networks and trajectories.

    This model jointly learns representations for road segments (nodes) and
    trajectories (sequences) using contrastive learning with three objectives:
    - Node-to-node: Same node across augmented views
    - Trajectory-to-trajectory: Same trajectory across augmented views
    - Node-to-trajectory: Nodes appearing in trajectories

    Args:
        vocab_size: Number of road segments (nodes)
        embed_size: Embedding dimension
        hidden_size: Hidden layer dimension
        edge_index: Graph edge indices tensor
        graph_encoder: GraphEncoder module
        seq_encoder: TransformerModel module
        mode: Encoding mode - 's' (structural) or 'p' (pure embeddings)
        device: Device for computation
    """

    def __init__(self, vocab_size, embed_size, hidden_size, edge_index,
                 graph_encoder, seq_encoder, mode='s', device=None):
        super(SingleViewModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.device = device if device is not None else torch.device('cpu')

        self.register_buffer('edge_index', edge_index)
        self.node_embedding = nn.Embedding(vocab_size, embed_size)
        self.padding = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=False)
        self.graph_encoder = graph_encoder
        self.seq_encoder = seq_encoder

    def encode_graph(self, drop_rate=0.):
        """Encode graph nodes using GNN with edge dropout.

        Args:
            drop_rate: Edge dropout rate for augmentation

        Returns:
            Node encodings (num_nodes, hidden_size)
        """
        node_emb = self.node_embedding.weight
        if HAS_PYGEOMETRIC and drop_rate > 0:
            edge_index = dropout_adj(self.edge_index, p=drop_rate)[0]
        else:
            edge_index = self.edge_index
        node_enc = self.graph_encoder(node_emb, edge_index)
        return node_enc

    def encode_sequence(self, sequences, drop_rate=0.):
        """Encode trajectory sequences using transformer.

        Args:
            sequences: Trajectory sequences (batch_size, max_seq_len)
            drop_rate: Masking rate for augmentation

        Returns:
            Trajectory encodings (batch_size, hidden_size)
        """
        if self.mode == 'p':
            # Pure embedding mode - use raw embeddings
            lookup_table = torch.cat([self.node_embedding.weight, self.padding], 0)
        else:
            # Structural mode - use GNN-encoded embeddings
            node_enc = self.encode_graph()
            lookup_table = torch.cat([node_enc, self.padding], 0)

        batch_size, max_seq_len = sequences.size()

        # Apply random masking for augmentation
        if drop_rate > 0:
            sequences = random_mask(sequences, self.vocab_size, drop_rate)

        # Create padding mask
        src_key_padding_mask = (sequences == self.vocab_size)
        pool_mask = (1 - src_key_padding_mask.int()).transpose(0, 1).unsqueeze(-1).float()

        # Lookup embeddings and encode
        seq_emb = torch.index_select(
            lookup_table, 0, sequences.view(-1)
        ).view(batch_size, max_seq_len, -1).transpose(0, 1)

        seq_enc = self.seq_encoder(seq_emb, None, src_key_padding_mask)

        # Mean pooling over valid positions
        pool_sum = pool_mask.sum(0)
        pool_sum = torch.clamp(pool_sum, min=1e-9)
        seq_pooled = (seq_enc * pool_mask).sum(0) / pool_sum

        return seq_pooled

    def forward(self, sequences, drop_edge_rate=0., drop_road_rate=0.):
        """Forward pass computing both node and sequence representations.

        Args:
            sequences: Trajectory sequences (batch_size, max_seq_len)
            drop_edge_rate: Edge dropout rate
            drop_road_rate: Road masking rate

        Returns:
            Tuple of (node_rep, seq_rep)
        """
        node_rep = self.encode_graph(drop_edge_rate)
        seq_rep = self.encode_sequence(sequences, drop_road_rate)
        return node_rep, seq_rep

    def get_node_embeddings(self):
        """Get final node embeddings.

        Returns:
            Node embeddings (num_nodes, hidden_size)
        """
        if self.mode == 'p':
            return self.node_embedding.weight
        else:
            return self.encode_graph(drop_rate=0.)

    def get_trajectory_embedding(self, sequences):
        """Get trajectory embeddings for given sequences.

        Args:
            sequences: Trajectory sequences (batch_size, max_seq_len)

        Returns:
            Trajectory embeddings (batch_size, hidden_size)
        """
        return self.encode_sequence(sequences, drop_rate=0.)


class JCLRNT(AbstractModel):
    """JCLRNT: Joint Contrastive Learning for Road Network and Trajectory.

    This is the main model class adapted for LibCity framework. It wraps the
    SingleViewModel and provides the standard LibCity interface.

    The model learns joint representations for road segments and trajectories
    using three contrastive objectives:
    1. Node-to-node (ss): Same node in two augmented graph views
    2. Trajectory-to-trajectory (tt): Same trajectory in two augmented views
    3. Node-to-trajectory (st): Nodes that appear in trajectories

    Args:
        config: Configuration dictionary
        data_feature: Dictionary containing data features like num_nodes, edge_index
    """

    def __init__(self, config, data_feature):
        super(JCLRNT, self).__init__(config, data_feature)

        self.config = config
        self.data_feature = data_feature

        # Model dimensions
        self.embed_size = config.get('embed_size', 128)
        self.hidden_size = config.get('hidden_size', 128)
        self.num_heads = config.get('num_heads', 4)
        self.num_transformer_layers = config.get('num_transformer_layers', 2)
        self.num_graph_layers = config.get('num_graph_layers', 2)

        # Dropout rates
        self.drop_rate = config.get('drop_rate', 0.2)
        self.drop_edge_rate = config.get('drop_edge_rate', 0.2)
        self.drop_road_rate = config.get('drop_road_rate', 0.2)

        # Loss configuration
        self.lambda_st = config.get('lambda_st', 0.8)
        self.loss_measure = config.get('loss_measure', 'jsd')
        self.weighted_loss = config.get('weighted_loss', False)

        # Encoding mode: 's' for structural (GNN), 'p' for pure embeddings
        self.mode = config.get('mode', 's')

        # Activation function
        activation_name = config.get('activation', 'relu')
        self.activation = nn.ReLU() if activation_name == 'relu' else nn.PReLU()

        # Device
        self.device = config.get('device', torch.device('cpu'))

        # Data features
        self.num_nodes = data_feature.get('num_nodes',
                                          data_feature.get('vocab_size',
                                                           data_feature.get('loc_size', 1000)))

        # Logger (initialize early so it can be used in edge_index check)
        self._logger = getLogger()

        # Edge index from data_feature
        edge_index = data_feature.get('edge_index', None)
        if edge_index is None:
            # Create self-loop edges as fallback
            self._logger.warning("No edge_index provided, creating self-loop graph")
            edge_index = torch.stack([
                torch.arange(self.num_nodes),
                torch.arange(self.num_nodes)
            ], dim=0)
        elif isinstance(edge_index, (list, tuple)):
            edge_index = torch.tensor(edge_index)

        # Log model configuration
        self._logger.info("Building JCLRNT model")
        self._logger.info(f"  num_nodes: {self.num_nodes}")
        self._logger.info(f"  embed_size: {self.embed_size}")
        self._logger.info(f"  hidden_size: {self.hidden_size}")
        self._logger.info(f"  mode: {self.mode}")
        self._logger.info(f"  loss_measure: {self.loss_measure}")
        self._logger.info(f"  lambda_st: {self.lambda_st}")

        # Build sub-modules
        if HAS_PYGEOMETRIC:
            graph_encoder = GraphEncoder(
                self.embed_size, self.hidden_size, GATConv,
                self.num_graph_layers, self.activation
            )
        else:
            self._logger.warning("PyTorch Geometric not available, using linear graph encoder")
            graph_encoder = GraphEncoder(
                self.embed_size, self.hidden_size, None,
                self.num_graph_layers, self.activation
            )

        seq_encoder = TransformerModel(
            self.hidden_size, self.num_heads, self.hidden_size,
            self.num_transformer_layers, self.drop_rate
        )

        # Main model
        self.model = SingleViewModel(
            vocab_size=self.num_nodes,
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
            edge_index=edge_index,
            graph_encoder=graph_encoder,
            seq_encoder=seq_encoder,
            mode=self.mode,
            device=self.device
        )

        # Loss weight computation
        # l_st = lambda_st, l_ss = l_tt = 0.5 * (1 - lambda_st)
        self.l_st = self.lambda_st
        self.l_ss = 0.5 * (1 - self.lambda_st)
        self.l_tt = 0.5 * (1 - self.lambda_st)

    def forward(self, batch):
        """Forward pass through the model.

        Args:
            batch: Dictionary containing:
                - 'current_loc': Current location sequences (batch_size, seq_len) with location IDs
                  (LibCity TrajectoryDataset format)
                - Or 'X': Trajectory sequences (batch_size, max_seq_len) for backward compatibility

        Returns:
            Dictionary with node and trajectory representations
        """
        # Handle different input formats - LibCity TrajectoryDataset uses 'current_loc'
        # Use try/except instead of 'in' operator since LibCity's Batch class lacks __contains__
        try:
            X = batch['current_loc']
        except KeyError:
            X = batch['X']

        # If X has 3 dimensions, take the first feature as road segment index
        if X.dim() == 3:
            sequences = X[:, :, 0].long()
        else:
            sequences = X.long()

        sequences = sequences.to(self.device)

        # Get representations
        node_rep, seq_rep = self.model(
            sequences,
            drop_edge_rate=self.drop_edge_rate if self.training else 0.,
            drop_road_rate=self.drop_road_rate if self.training else 0.
        )

        return {
            'node_rep': node_rep,
            'seq_rep': seq_rep,
            'sequences': sequences
        }

    def predict(self, batch):
        """Generate embeddings for a batch.

        For representation learning models, predict returns the learned embeddings.

        Args:
            batch: Dictionary containing input data

        Returns:
            Dictionary with trajectory embeddings
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(batch)
            return {
                'trajectory_embedding': result['seq_rep'],
                'node_embedding': result['node_rep']
            }

    def calculate_loss(self, batch):
        """Calculate combined contrastive loss.

        The loss combines three contrastive objectives:
        1. Node-to-node loss (ss): Same nodes across augmented graphs
        2. Trajectory-to-trajectory loss (tt): Same trajectories across augmented views
        3. Node-to-trajectory loss (st): Nodes appearing in trajectories

        Total loss = l_ss * loss_ss + l_tt * loss_tt + l_st * loss_st
        where l_st = lambda_st, l_ss = l_tt = 0.5 * (1 - lambda_st)

        Args:
            batch: Dictionary containing input data and optional weights

        Returns:
            Total loss tensor
        """
        # Get sequences - LibCity TrajectoryDataset uses 'current_loc'
        # Use try/except instead of 'in' operator since LibCity's Batch class lacks __contains__
        try:
            X = batch['current_loc']
        except KeyError:
            X = batch['X']
        if X.dim() == 3:
            sequences = X[:, :, 0].long()
        else:
            sequences = X.long()
        sequences = sequences.to(self.device)

        # Forward pass with augmentation - view 1
        node_rep1, seq_rep1 = self.model(
            sequences,
            drop_edge_rate=self.drop_edge_rate,
            drop_road_rate=self.drop_road_rate
        )

        # Forward pass with augmentation - view 2
        node_rep2, seq_rep2 = self.model(
            sequences,
            drop_edge_rate=self.drop_edge_rate,
            drop_road_rate=self.drop_road_rate
        )

        # Node-to-node contrastive loss
        loss_ss = node_node_loss(node_rep1, node_rep2, self.loss_measure, self.device)

        # Trajectory-to-trajectory contrastive loss
        loss_tt = seq_seq_loss(seq_rep1, seq_rep2, self.loss_measure, self.device)

        # Node-to-trajectory contrastive loss
        # Use try/except instead of 'in' operator since LibCity's Batch class lacks __contains__
        weights = None
        try:
            if self.weighted_loss:
                weights = batch['weights'].to(self.device)
        except KeyError:
            weights = None

        if weights is not None:
            loss_st1 = weighted_ns_loss(node_rep1, seq_rep2, weights, self.loss_measure)
            loss_st2 = weighted_ns_loss(node_rep2, seq_rep1, weights, self.loss_measure)
        else:
            loss_st1 = node_seq_loss(
                node_rep1, seq_rep2, sequences,
                self.loss_measure, self.device, self.num_nodes
            )
            loss_st2 = node_seq_loss(
                node_rep2, seq_rep1, sequences,
                self.loss_measure, self.device, self.num_nodes
            )
        loss_st = (loss_st1 + loss_st2) / 2

        # Combined loss
        total_loss = self.l_ss * loss_ss + self.l_tt * loss_tt + self.l_st * loss_st

        return total_loss

    def get_node_embeddings(self):
        """Get learned node (road segment) embeddings.

        Returns:
            Node embeddings tensor (num_nodes, hidden_size)
        """
        self.eval()
        with torch.no_grad():
            return self.model.get_node_embeddings()

    def get_trajectory_embedding(self, sequences):
        """Get trajectory embedding for given sequences.

        Args:
            sequences: Trajectory sequences (batch_size, max_seq_len)

        Returns:
            Trajectory embeddings (batch_size, hidden_size)
        """
        self.eval()
        with torch.no_grad():
            if isinstance(sequences, list):
                sequences = torch.tensor(sequences)
            sequences = sequences.to(self.device)
            return self.model.get_trajectory_embedding(sequences)
