"""
Backbone encoders for BEACON.

Produces continuous sequence representations from one-hot encoded DNA.
Supports both standard convolutional and dilated convolutional architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """Residual convolutional block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = F.gelu(x)
        return x


class ConvBackbone(nn.Module):
    """
    Standard convolutional backbone encoder.

    Architecture inspired by BPNet but adapted for slot attention.
    Input: [B, L, 4] one-hot encoded DNA
    Output: [B, L, D] continuous sequence representation
    """

    def __init__(
        self,
        input_channels: int = 4,
        hidden_dim: int = 256,
        output_dim: int = 256,
        n_layers: int = 6,
        kernel_size: int = 11,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initial projection
        self.input_proj = nn.Conv1d(input_channels, hidden_dim, kernel_size=1)

        # Convolutional layers
        self.layers = nn.ModuleList([
            ResidualBlock(hidden_dim, kernel_size=kernel_size, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: One-hot encoded sequence [B, L, 4]

        Returns:
            Sequence representation [B, L, D]
        """
        # [B, L, 4] -> [B, 4, L]
        x = x.transpose(1, 2)

        # Initial projection
        x = self.input_proj(x)

        # Apply convolutional layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        x = self.output_proj(x)

        # [B, D, L] -> [B, L, D]
        x = x.transpose(1, 2)

        # Layer norm
        x = self.layer_norm(x)

        return x


class DilatedConvBackbone(nn.Module):
    """
    Dilated convolutional backbone with exponentially increasing dilation.

    Provides larger receptive field for capturing long-range dependencies
    while maintaining resolution. Similar to WaveNet/BPNet architecture.

    Input: [B, L, 4] one-hot encoded DNA
    Output: [B, L, D] continuous sequence representation
    """

    def __init__(
        self,
        input_channels: int = 4,
        hidden_dim: int = 256,
        output_dim: int = 256,
        n_layers: int = 9,
        kernel_size: int = 3,
        max_dilation: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initial wide convolution to capture local patterns
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=21, padding=10),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Dilated residual blocks with exponentially increasing dilation
        self.dilated_blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = min(2 ** i, max_dilation)
            self.dilated_blocks.append(
                ResidualBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1),
            nn.GELU(),
        )
        self.layer_norm = nn.LayerNorm(output_dim)

        # Calculate receptive field
        self._receptive_field = self._calculate_receptive_field(n_layers, kernel_size, max_dilation)

    def _calculate_receptive_field(self, n_layers: int, kernel_size: int, max_dilation: int) -> int:
        """Calculate the receptive field of the network."""
        rf = 21  # Initial convolution
        for i in range(n_layers):
            dilation = min(2 ** i, max_dilation)
            rf += 2 * (kernel_size - 1) * dilation
        return rf

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: One-hot encoded sequence [B, L, 4]

        Returns:
            Sequence representation [B, L, D]
        """
        # [B, L, 4] -> [B, 4, L]
        x = x.transpose(1, 2)

        # Initial convolution
        x = self.input_conv(x)

        # Dilated blocks
        for block in self.dilated_blocks:
            x = block(x)

        # Output projection
        x = self.output_proj(x)

        # [B, D, L] -> [B, L, D]
        x = x.transpose(1, 2)

        # Layer norm
        x = self.layer_norm(x)

        return x


class ConvBackboneWithMultiScale(nn.Module):
    """
    Multi-scale convolutional backbone.

    Captures patterns at multiple scales using parallel convolutions
    with different kernel sizes, then fuses the representations.
    """

    def __init__(
        self,
        input_channels: int = 4,
        hidden_dim: int = 256,
        output_dim: int = 256,
        kernel_sizes: List[int] = [3, 7, 11, 21],
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_scales = len(kernel_sizes)
        scale_dim = hidden_dim // self.n_scales

        # Multi-scale input convolutions
        self.input_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, scale_dim, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(scale_dim),
                nn.GELU(),
            )
            for ks in kernel_sizes
        ])

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.layers = nn.ModuleList([
            ResidualBlock(hidden_dim, kernel_size=7, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, 4] -> [B, 4, L]
        x = x.transpose(1, 2)

        # Multi-scale convolutions
        scale_outputs = [conv(x) for conv in self.input_convs]

        # Concatenate scales
        x = torch.cat(scale_outputs, dim=1)

        # Fusion
        x = self.fusion(x)

        # Residual layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        x = self.output_proj(x)

        # [B, D, L] -> [B, L, D]
        x = x.transpose(1, 2)
        x = self.layer_norm(x)

        return x
