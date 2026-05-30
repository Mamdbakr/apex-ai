"""
Deep sequence encoders for pose data.

Three architectures are provided; pick whichever fits the deployment
constraint:

    - PoseLSTM        : low-param, easy to train, sequential inference is fast
    - PoseTCN         : causal dilated 1D convs, parallel training, low latency
    - PoseTransformer : best accuracy on long sequences, higher compute

All three share the same I/O contract:

    Input  : (B, T, F) float tensor
             B = batch, T = time steps, F = features per frame
             F = 142 matches `SequenceBuffer.feature_tensor()`
    Output : (B, num_classes) logits

They are exercise/quality classifiers — train them on labelled clips of
(exercise, form_quality) and run them on the live SequenceBuffer.

PyTorch is imported lazily so apex_ml works without torch installed for
users who only want the rule-based feedback engine.
"""

from __future__ import annotations

import math
from typing import Optional

# Lazy torch import — raises a clear error only if these models are used
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:                              # pragma: no cover
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "apex_ml.sequence_models requires PyTorch. "
            "Install with `pip install torch`."
        )


# =============================================================== LSTM
class PoseLSTM(nn.Module if _TORCH_AVAILABLE else object):
    """Stacked LSTM encoder for pose sequences.

    Two layers with dropout, then a temporal attention pool over hidden
    states (more robust than just using the final hidden state — partial
    reps don't necessarily end at an informative timestep).
    """

    def __init__(self, input_dim: int = 142, hidden_dim: int = 128,
                 num_layers: int = 2, num_classes: int = 5,
                 dropout: float = 0.3, bidirectional: bool = True):
        _require_torch()
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attn = nn.Linear(out_dim, 1)
        self.norm = nn.LayerNorm(out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):                                          # (B, T, F)
        h, _ = self.lstm(x)                                        # (B, T, H')
        # Attention pooling over time
        w = torch.softmax(self.attn(h), dim=1)                     # (B, T, 1)
        pooled = (h * w).sum(dim=1)                                # (B, H')
        pooled = self.norm(pooled)
        return self.head(pooled)                                   # (B, C)


# =============================================================== TCN
class _CausalConv1d(nn.Module if _TORCH_AVAILABLE else object):
    """1D causal convolution — uses only past + current samples.

    Critical for live inference: we never want to peek at future frames.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        _require_torch()
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):                                          # (B, C, T)
        x = F.pad(x, (self.pad, 0))                                # left-pad only
        return self.conv(x)


class _TCNBlock(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float):
        _require_torch()
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            _CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return F.relu(self.net(x) + self.proj(x))


class PoseTCN(nn.Module if _TORCH_AVAILABLE else object):
    """Temporal Convolutional Network with exponentially dilated layers.

    Receptive field grows as 2^L, so 6 layers of kernel 3 cover ~127
    timesteps — comfortably enough for one rep at 30 FPS.
    """

    def __init__(self, input_dim: int = 142, channels: int = 96,
                 num_layers: int = 6, kernel_size: int = 3,
                 num_classes: int = 5, dropout: float = 0.2):
        _require_torch()
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            layers.append(_TCNBlock(in_ch, channels, kernel_size,
                                    dilation=2 ** i, dropout=dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x):                                          # (B, T, F)
        x = x.transpose(1, 2)                                      # (B, F, T)
        x = self.tcn(x)                                            # (B, C, T)
        return self.head(x)                                        # (B, num_classes)


# ======================================================= Transformer
class _PositionalEncoding(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, d_model: int, max_len: int = 512):
        _require_torch()
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                              * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))                # (1, L, D)

    def forward(self, x):                                          # (B, T, D)
        return x + self.pe[:, : x.size(1)]


class PoseTransformer(nn.Module if _TORCH_AVAILABLE else object):
    """Transformer encoder for pose sequences.

    A [CLS] token aggregates the sequence into a single classification
    embedding (same trick as BERT). For live inference, attention is
    masked causally so we can run it on a sliding window without future
    leakage.
    """

    def __init__(self, input_dim: int = 142, d_model: int = 128,
                 num_heads: int = 4, num_layers: int = 3,
                 num_classes: int = 5, dropout: float = 0.1,
                 causal: bool = True, max_len: int = 256):
        _require_torch()
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = _PositionalEncoding(d_model, max_len=max_len + 1)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)
        self.causal = causal

    def _causal_mask(self, T: int, device) -> "torch.Tensor":
        # Upper-triangular -inf mask (no peeking at the future)
        return torch.triu(torch.ones(T, T, device=device) * float("-inf"), diagonal=1)

    def forward(self, x):                                          # (B, T, F)
        B, T, _ = x.shape
        x = self.proj(x)                                           # (B, T, D)
        cls = self.cls.expand(B, -1, -1)                           # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                             # (B, T+1, D)
        x = self.pos(x)
        mask = self._causal_mask(T + 1, x.device) if self.causal else None
        z = self.encoder(x, mask=mask)
        return self.head(z[:, 0])                                  # (B, C) from CLS


# ============================================================ Helpers
def build_model(name: str, **kwargs):
    """Factory: 'lstm' | 'tcn' | 'transformer' -> instantiated model."""
    _require_torch()
    name = name.lower()
    if name == "lstm":
        return PoseLSTM(**kwargs)
    if name == "tcn":
        return PoseTCN(**kwargs)
    if name == "transformer":
        return PoseTransformer(**kwargs)
    raise ValueError(f"Unknown model: {name!r}. Choose lstm|tcn|transformer.")
