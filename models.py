from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", output_dim=256):
        super().__init__()
        self.device = device
        self.repr_dim = output_dim

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        B, T, _ = actions.shape

        return torch.randn((B, T + 1, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output



def conv3x3(in_c, out_c, stride=1, padding=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=padding, bias=False)


class Encoder(nn.Module):
    """
    Final output shape: (B, 50, 8, 8)

    Configuration:
    - Conv1: 2 -> 24, kernel_size=5, stride=2, padding=2 -> (B,24,33,33)
      BN + LeakyReLU
    - Conv2: 24 -> 50, kernel_size=3, stride=2, padding=1 -> (B,50,17,17)
      BN + LeakyReLU
    - AdaptiveMaxPool2d((8,8)) -> (B,50,8,8)
    """
    def __init__(self, input_channels=2, hidden_channels=24, final_channels=50):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        self.conv2 = nn.Conv2d(hidden_channels, final_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(final_channels)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        self.pool = nn.AdaptiveMaxPool2d((8,8))  # (B,50,8,8)

    def forward(self, x):
        # x: (B,2,65,65)
        x = self.conv1(x)         # (B,24,33,33)
        x = self.bn1(x)
        x = self.leaky_relu1(x)

        x = self.conv2(x)         # (B,50,17,17)
        x = self.bn2(x)
        x = self.leaky_relu2(x)

        x = self.pool(x)          # (B,50,8,8)
        return x


class ActionProjector(nn.Module):
    """
    Encodes and tiles actions into spatial planes: (B,2,8,8)
    """
    def __init__(self, H=8, W=8):
        super().__init__()
        self.H = H
        self.W = W

    def forward(self, actions):
        dx = actions[:, 0]
        dy = actions[:, 1]

        # Example normalization (domain-specific)
        dx = (dx + 1.7997442) / (2 * 1.7997442)
        dy = (dy + 1.7997442) / (2 * 1.7997442)

        dx_plane = dx.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.H, self.W) 
        dy_plane = dy.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.H, self.W)
        action_planes = torch.cat([dx_plane, dy_plane], dim=1)  # (B,2,8,8)

        return action_planes


class Predictor(nn.Module):
    """
    Final output: (B,50,8,8).

    Input shape:
      - encoder_output: (B,50,8,8)
      - action_planes: (B,2,8,8)
    => concatenate -> (B,52,8,8)

    Layers:
      - conv1: 52 -> 24
      - conv2: 24 -> 50
      - Residual add => final shape (B,50,8,8)
    """
    def __init__(self, encoder_out_ch=50, hidden_channels=24, final_channels=50):
        super().__init__()
        self.action_projector = ActionProjector(H=8, W=8)
        
        self.input_channels = encoder_out_ch + 2  # 50 + 2 = 52
        self.conv1 = conv3x3(self.input_channels, hidden_channels, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        self.conv2 = conv3x3(hidden_channels, final_channels, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(final_channels)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        # Skip connection if input_channels != final_channels
        if self.input_channels != final_channels:
            self.residual_proj = conv3x3(self.input_channels, final_channels, stride=1, padding=1)
            self.bn_residual = nn.BatchNorm2d(final_channels)
        else:
            self.residual_proj = None
    
    def forward(self, encoder_output, actions):
        """
        encoder_output: (B,50,8,8)
        actions: (B,2)
        Final predictor output shape: (B,50,8,8)
        """
        action_planes = self.action_projector(actions)  # (B,2,8,8)
        x = torch.cat([encoder_output, action_planes], dim=1)  # (B,52,8,8)

        residual = x
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
            residual = self.bn_residual(residual)

        x = self.conv1(x)    # (B,24,8,8)
        x = self.bn1(x)
        x = self.leaky_relu1(x)

        x = self.conv2(x)    # (B,50,8,8)
        x = self.bn2(x)
        x = self.leaky_relu2(x)

        x += residual        # (B,50,8,8)
        return x
    



class JEPA_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.predictor = Predictor()
        self.repr_dim = 3200


    def forward(self, states, actions):

        B, T = states.size(0), actions.size(1)+1

        s_n = self.encoder(states[:, 0])  # given the initial state from the target to be fair

        preds = [s_n]
        for n in range(1,T):
            
            s_n = self.predictor(s_n, actions[:, n-1]) # (B,64,4,4)
            preds.append(s_n)
        s_n_pred_all = torch.stack(preds, dim=1) # (B,T,64,4,4)

        return s_n_pred_all