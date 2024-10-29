import torch
import torch.nn as nn


class GatedEquivariantBlock(nn.Module):
    """
    Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """
    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            nn.SiLU(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = nn.SiLU() if scalar_activation else None
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)
    
    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v
    
    # def forward(self, x, v):
    #     assert v.shape[1] == 3 or v.shape[1] == 8, "v must be of shape (N, 3, *) or (N, 8, *)"
    #     vec1 = self.vec1_proj(v)
    #     vec2 = self.vec2_proj(v)
        
    #     if v.shape[1] == 8:
    #         l1_vec1, l2_vec1 = torch.split(vec1, [3, 5], dim=1)
    #         norm = torch.norm(l1_vec1, dim=1) + torch.norm(l2_vec1, dim=1)
    #     else:
    #         norm = torch.norm(vec1, dim=1)

    #     x = torch.cat([x, norm], dim=-1)
    #     x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
    #     v = v.unsqueeze(1) * vec2

    #     if self.act is not None:
    #         x = self.act(x)
    #     return x, v


class EquivariantScalar(nn.Module):
    def __init__(self, hidden_channels):
        super(EquivariantScalar, self).__init__()
        self.output_network = nn.ModuleList([
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2, 
                    1, 
                    scalar_activation=False,
                ),
        ])
        
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()
    
    def forward(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        return x + v.sum() * 0
    
class Scalar(nn.Module):
    def __init__(self, hidden_channels):
        super(Scalar, self).__init__()
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def forward(self, x):
        return self.output_network(x)