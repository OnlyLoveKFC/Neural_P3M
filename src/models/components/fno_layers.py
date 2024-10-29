# This code is modified from the `neuraloperator` package
# https://github.com/neuraloperator/neuraloperator
# THIS CODE IS HARDCODED FOR 3D GRID DATA

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """A Multi-Layer Perceptron, with arbitrary number of layers

    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is F.gelu
    dropout : float, default is 0
        if > 0, dropout probability
    """

    def __init__(
            self,
            in_channels,
            out_channels=None,
            hidden_channels=None,
            n_layers=2,
            n_dim=2,
            non_linearity=F.silu,
            dropout=0.0,
            **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            if dropout > 0.0
            else None
        )
        Conv = getattr(nn, f"Conv{n_dim}d")
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                self.fcs.append(Conv(self.in_channels, self.out_channels, 1))
            elif i == 0:
                self.fcs.append(Conv(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(Conv(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(Conv(self.hidden_channels, self.hidden_channels, 1))
        
        self.reset_parameters()
                
    def reset_parameters(self):
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        return x


class SoftGating(nn.Module):
    def __init__(self, in_features, out_features=None, n_dim=2, bias=False):
        super().__init__()
        if out_features is not None and in_features != out_features:
            raise ValueError(
                f"Got in_features={in_features} and out_features={out_features}"
                "but these two must be the same for soft-gating"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        if bias:
            self.bias = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        else:
            self.bias = None
            
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.ones_(self.bias)

    def forward(self, x):
        """Applies soft-gating to a batch of activations"""
        if self.bias is not None:
            return self.weight * x + self.bias
        else:
            return self.weight * x


class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes, n_layers=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # n_modes is the total number of modes kept along each dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)
        self.max_n_modes = self.n_modes
        self.n_layers = n_layers

        self.weight = nn.ParameterList([
            nn.Parameter(
                torch.view_as_real(torch.empty(in_channels, out_channels, *self.max_n_modes, dtype=torch.cfloat))
            )
            for _ in range(n_layers)
        ])
        self.bias = nn.Parameter(
            torch.empty(*((n_layers, self.out_channels) + (1,) * self.order))
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weight:
            nn.init.normal_(w, 0, (2 / (self.in_channels + self.out_channels)) ** 0.5)
        nn.init.normal_(self.bias, 0, (2 / (self.in_channels + self.out_channels)) ** 0.5)

    def _get_weight(self, index):
        return torch.view_as_complex(self.weight[index])

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        n_modes = list(n_modes)
        # The last mode has a redundacy as we use real FFT
        # As a design choice we do the operation here to avoid users dealing with the +1
        n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def forward(self, x: torch.Tensor, indices=0):
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient
        fft_dims = list(range(-self.order, 0))

        x = torch.fft.rfftn(x, norm='backward', dim=fft_dims)
        if self.order > 1:
            x = torch.fft.fftshift(x, dim=fft_dims[:-1])

        out_fft = torch.zeros([batchsize, self.out_channels, *fft_size], device=x.device, dtype=torch.cfloat)
        starts = [
            (max_modes - min(size, n_mode)) for (size, n_mode, max_modes) in
            zip(fft_size, self.n_modes, self.max_n_modes)
        ]
        slices_w = [slice(None), slice(None)]  # Batch_size, channels
        slices_w += [slice(start // 2, -start // 2) if start else slice(start, None) for start in starts[:-1]]
        # The last mode already has redundant half removed
        slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
        weight = self._get_weight(indices)[slices_w]

        starts = [(size - min(size, n_mode)) for (size, n_mode) in zip(list(x.shape[2:]), list(weight.shape[2:]))]
        slices_x = [slice(None), slice(None)]  # Batch_size, channels
        slices_x += [slice(start // 2, -start // 2) if start else slice(start, None) for start in starts[:-1]]
        # The last mode already has redundant half removed
        slices_x += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
        out_fft[slices_x] = torch.einsum("bcxyz,cdxyz->bdxyz", x[slices_x], weight)

        if self.order > 1:
            out_fft = torch.fft.ifftshift(out_fft, dim=fft_dims[:-1])
        x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm='backward')
        x = x + self.bias[indices, ...]
        return x


class FNOBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes, n_layers=1, non_linearity=F.silu):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity

        self.convs = SpectralConv(
            self.in_channels,
            self.out_channels,
            self.n_modes,
            n_layers=n_layers,
        )

        self.fno_skips = nn.ModuleList(
            [
                SoftGating(
                    self.in_channels,
                    self.out_channels,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.convs.reset_parameters()
        for fno_skip in self.fno_skips:
            fno_skip.reset_parameters()

    def forward(self, x, index=0):
        x_skip_fno = self.fno_skips[index](x)
        x_fno = self.convs(x, index)
        x = x_fno + x_skip_fno
        if index < (self.n_layers - 1):
            x = self.non_linearity(x)
        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.convs.n_modes = n_modes
        self._n_modes = n_modes
