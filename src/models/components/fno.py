# This code is modified from the `neuraloperator` package
# https://github.com/neuraloperator/neuraloperator
# THIS CODE IS HARDCODED FOR 3D GRID DATA

import torch.nn as nn
import torch.nn.functional as F

from .fno_layers import MLP, FNOBlocks


class FNO(nn.Module):
    def __init__(
            self,
            n_modes,
            hidden_channels,
            in_channels=1,
            out_channels=1,
            n_layers=1,
            lifting_channels=256,
            projection_channels=256,
            non_linearity=F.silu,
    ):
        super().__init__()
        self.n_dim = len(n_modes)

        self._n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity

        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            n_layers=n_layers,
            non_linearity=non_linearity,
        )

        # if lifting_channels is passed, make lifting an MLP
        # with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
            )
        # otherwise, make it a linear layer
        else:
            self.lifting = MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
            )
        self.projection = MLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lifting.reset_parameters()
        self.fno_blocks.reset_parameters()
        self.projection.reset_parameters()

    def forward(self, x):
        x = self.lifting(x)
        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx)
        x = self.projection(x)
        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes


class FNO3d(FNO):
    def __init__(
            self,
            n_modes_height,
            n_modes_width,
            n_modes_depth,
            hidden_channels,
            in_channels=1,
            out_channels=1,
            n_layers=1,
            lifting_channels=256,
            projection_channels=256,
            non_linearity=F.silu,
    ):
        super().__init__(
            n_modes=(n_modes_height, n_modes_width, n_modes_depth),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            non_linearity=non_linearity,
        )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width
        self.n_modes_depth = n_modes_depth
