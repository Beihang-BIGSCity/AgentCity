"""
WordVocab class for START model.
Handles vocabulary building and encoding for trajectory data.

Migrated from: /home/wangwenrui/shk/AgentCity/repos/START/libcity/data/dataset/bert_vocab.py
"""

import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp
from collections import Counter
from tqdm import tqdm


def _calculate_normalized_laplacian(adj):
    """Calculate normalized Laplacian matrix."""
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def _calculate_random_walk_laplacian(adj_mx):
    """Calculate random walk Laplacian matrix."""
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


class WordVocab:
    """
    Vocabulary class for handling location tokens in trajectory data.

    This class builds a vocabulary from trajectory data and provides
    methods for converting between location IDs and vocabulary indices.

    Attributes:
        pad_index (int): Index for padding token
        unk_index (int): Index for unknown token
        sos_index (int): Index for start of sequence token (CLS)
        mask_index (int): Index for mask token (if use_mask=True)
        sep_index (int): Index for end of sequence token (if eos=True)
        vocab_size (int): Total vocabulary size
        user_num (int): Number of unique users
    """

    def __init__(self, roadmap_path, traj_path, min_freq=1, use_mask=True, seq_len=128, eos=False):
        """
        Initialize WordVocab from trajectory data.

        Args:
            roadmap_path: Path to road map file (currently unused but kept for compatibility)
            traj_path: Path to trajectory CSV file
            min_freq: Minimum frequency for a location to be included in vocabulary
            use_mask: Whether to include mask token in vocabulary
            seq_len: Maximum sequence length for processing
            eos: Whether to include end of sequence token
        """
        self.pad_index = 0
        self.unk_index = 1
        self.sos_index = 2  # CLS
        if use_mask:
            self.mask_index = 3
            specials = ["<pad>", "<unk>", "<sos>", "<mask>"]
            if eos:
                self.sep_index = 4
                specials = ["<pad>", "<unk>", "<sos>", "<mask>", "<eos>"]
        else:
            specials = ["<pad>", "<unk>", "<sos>"]

        train = pd.read_csv(traj_path, sep=';')

        counter = Counter()
        paths = train['path'].values
        for i in tqdm(range(paths.shape[0]), desc='location counting'):
            path_l = eval(paths[i])[:seq_len]
            for p in path_l:
                counter[p] += 1
        self.freqs = counter
        min_freq = max(min_freq, 1)

        self.index2loc = list(specials)
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        self.del_edge = []
        self.all_edge = []
        for word, freq in words_and_frequencies:
            if freq < min_freq:
                self.del_edge.append(word)
                continue
            self.index2loc.append(word)
            self.all_edge.append(word)
        self.loc2index = {tok: i for i, tok in enumerate(self.index2loc)}
        self.vocab_size = len(self.index2loc)

        users = train['usr_id'].values
        users = np.unique(users)
        specials = ["<pad>", "<unk>"]
        self.index2usr = list(specials)
        for u in users:
            self.index2usr.append(u)
        self.usr2index = {tok: i for i, tok in enumerate(self.index2usr)}
        self.user_num = len(self.index2usr)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        """
        Convert a sentence (list of locations) to a sequence of indices.

        Args:
            sentence: List of location IDs or space-separated string
            seq_len: Target sequence length (pad or truncate)
            with_eos: Whether to add EOS token (not implemented)
            with_sos: Whether to add SOS token at the beginning
            with_len: Whether to return original length

        Returns:
            Sequence of indices, optionally with original length
        """
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.loc2index.get(word, self.unk_index) for word in sentence]

        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        """
        Convert a sequence of indices back to locations.

        Args:
            seq: List of vocabulary indices
            join: Whether to join locations with spaces
            with_pad: Whether to include padding tokens

        Returns:
            List of location IDs or space-separated string
        """
        words = [self.index2loc[idx]
                 if idx < len(self.index2loc)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        """
        Load vocabulary from pickle file.

        Args:
            vocab_path: Path to vocabulary pickle file

        Returns:
            WordVocab instance
        """
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        """
        Save vocabulary to pickle file.

        Args:
            vocab_path: Path to save vocabulary
        """
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

    def __eq__(self, other):
        if self.loc2index != other.loc2index:
            return False
        if self.index2loc != other.index2loc:
            return False
        return True

    def __len__(self):
        return len(self.index2loc)

    def vocab_rerank(self):
        """Re-rank vocabulary indices based on current index2loc order."""
        self.loc2index = {word: i for i, word in enumerate(self.index2loc)}

    def extend(self, v, sort=False):
        """
        Extend vocabulary with words from another vocabulary.

        Args:
            v: Another WordVocab instance
            sort: Whether to sort new words alphabetically
        """
        words = sorted(v.index2loc) if sort else v.index2loc
        for w in words:
            if w not in self.loc2index:
                self.index2loc.append(w)
                self.loc2index[w] = len(self.index2loc) - 1
