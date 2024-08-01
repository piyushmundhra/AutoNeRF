import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from PIL import Image
from nerf import preprocess_data, volume_rendering, sample_along_rays

class MultiResHashEncoding(nn.Module):
    """
    MultiResHashEncoding is a neural network module that performs multi-resolution hash encoding.
    It takes an input tensor and encodes it into a set of features at different resolutions using hash tables.

    Args:
        n_levels (int): The number of levels or resolutions for encoding. Default is 16.
        n_features_per_level (int): The number of features per level. Default is 2.
        log2_hashmap_size (int): The logarithm base 2 of the hash map size. Default is 19.
        base_resolution (int): The base resolution for encoding. Default is 16.

    Methods:
        forward(x): (B,3) Performs forward pass of the input tensor through the encoding layers.
        hash(x): Computes the hash value for a given input tensor.
    """

    DEFAULT_CONFIG : dict = {
        'n_levels': 16,
        'n_features_per_level': 2,
        'log2_hashmap_size': 19,
        'base_resolution': 16
    }

    PRIMES = torch.tensor([1, 2654435761, 805459861])

    def __init__(self, n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution

        self.hash_tables = nn.ModuleList([
            nn.Embedding(2**log2_hashmap_size, n_features_per_level)
            for _ in range(n_levels)
        ])

        self.b = np.exp((np.log(2048) - np.log(base_resolution)) / (n_levels - 1))

    def nearest_points(floor, ceil):
        return torch.stack([
            torch.stack([floor[:, 0], floor[:, 1], floor[:, 2]], dim=-1),
            torch.stack([floor[:, 0], floor[:, 1], ceil[:, 2]], dim=-1),
            torch.stack([floor[:, 0], ceil[:, 1], floor[:, 2]], dim=-1),
            torch.stack([floor[:, 0], ceil[:, 1], ceil[:, 2]], dim=-1),
            torch.stack([ceil[:, 0], floor[:, 1], floor[:, 2]], dim=-1),
            torch.stack([ceil[:, 0], floor[:, 1], ceil[:, 2]], dim=-1),
            torch.stack([ceil[:, 0], ceil[:, 1], floor[:, 2]], dim=-1),
            torch.stack([ceil[:, 0], ceil[:, 1], ceil[:, 2]], dim=-1),
        ], dim=1)
    
    def weights(frac):
        return torch.stack([
            (1 - frac[:, 0]) * (1 - frac[:, 1]) * (1 - frac[:, 2]),
            (1 - frac[:, 0]) * (1 - frac[:, 1]) * frac[:, 2],
            (1 - frac[:, 0]) * frac[:, 1] * (1 - frac[:, 2]),
            (1 - frac[:, 0]) * frac[:, 1] * frac[:, 2],
            frac[:, 0] * (1 - frac[:, 1]) * (1 - frac[:, 2]),
            frac[:, 0] * (1 - frac[:, 1]) * frac[:, 2],
            frac[:, 0] * frac[:, 1] * (1 - frac[:, 2]),
            frac[:, 0] * frac[:, 1] * frac[:, 2],
        ], dim=1)

    def forward(self, x):
        encoded_features = []
        for i in range(self.n_levels):
            resolution = int(self.base_resolution * self.b**i)
            x_scaled = x * resolution
            floor = torch.floor(x_scaled).long()
            ceil = torch.ceil(x_scaled).long()
            frac = x_scaled - floor

            pts = MultiResHashEncoding.nearest_points(floor=floor, ceil=ceil)
            h = self.hash2(pts)

            embeds = self.hash_tables[i](h)
            weights = MultiResHashEncoding.weights(frac=frac)

            interpolated_embedding = torch.sum(embeds * weights.unsqueeze(-1), dim=1)

            encoded_features.append(interpolated_embedding)
        
        return torch.cat(encoded_features, dim=-1)

    def hash1(self, x):
        x = torch.sum(x * MultiResHashEncoding.PRIMES, dim=-1)
        return torch.bitwise_and(x, (2**self.log2_hashmap_size - 1))

    # from https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/blob/main/Instant_Neural_Graphics_Primitives_with_a_Multiresolution_Hash_Encoding/ngp.py
    def hash2(self, x):
        x = x * MultiResHashEncoding.PRIMES
        return torch.remainder(torch.bitwise_xor(torch.bitwise_xor(x[:, :, 0], x[:, :, 1]), x[:, :, 2]), 2**self.log2_hashmap_size)
    
class InstantNGP(nn.Module):
    """
    Neural Graphics Pipeline model for instant NeRF.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, 3) -- [[x,y,z]]

    Outputs:
        torch.Tensor: Output tensor of shape (batch_size, 4) -- [[R,G,B,A]]
    """

    def __init__(self, encoding_config=None, hidden_dim=64, num_layers=3):
        super().__init__()

        self.encoding = MultiResHashEncoding(**encoding_config) if encoding_config is not None else MultiResHashEncoding(**MultiResHashEncoding.DEFAULT_CONFIG)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.encoding.n_levels * self.encoding.n_features_per_level, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_layers - 2)],
            nn.Linear(hidden_dim, 4)  # 3 for RGB, 1 for density
        )

    def forward(self, x):
        encoded = self.encoding(x)
        output = self.mlp(encoded)
        rgb = torch.sigmoid(output[:, :3])  
        alpha = torch.sigmoid(output[:, 3:])  
        return torch.cat([rgb, alpha], dim=-1)
    
@torch.no_grad()
def infer(model: InstantNGP, data_loader: DataLoader):
    model.eval()  # Set the model to evaluation mode

    rgb_pred = None
    density = None
    depths = None
    
    for data in tqdm(data_loader, desc="Inference Progress", unit="batch"):
        B = data.shape[0]
        num_samples = 64
        near = 1.0
        far = 5.0

        origins: torch.Tensor = data[:, :3]
        directions: torch.Tensor = data[:, 3:6]
        positions = sample_along_rays(origins, directions, num_samples, near, far)
        _depths = positions[:, :, 2]
        positions = positions.reshape(shape=(positions.shape[0] * num_samples, 3))
        
        directions = directions.repeat_interleave(repeats=num_samples, dim=0)
        model_output = model.forward(x=positions)
        _rgb_pred = model_output[:, :3].reshape(B, num_samples, 3)
        _density = model_output[:, 3].reshape(B, num_samples, 1).squeeze(-1)
        
        if rgb_pred is None:
            rgb_pred = _rgb_pred
            density = _density
            depths = _depths
        else:
            rgb_pred = torch.concat((rgb_pred, _rgb_pred), 0)
            density = torch.concat((density, _density), 0)
            depths = torch.concat((depths, _depths), 0)
    
    print(rgb_pred.shape)
    print(density.shape)
    print(depths.shape)

    rendered_rgb = volume_rendering(rgb_pred, density, depths)
    rendered_rgb = (rendered_rgb * 255).clamp(0, 255).byte().reshape((518,518,3))
    image = Image.fromarray(rendered_rgb.numpy())

    return image

@torch.enable_grad()
def train(model: InstantNGP, data_loader: DataLoader):
    num_samples: int = 64
    near: float = 1.0
    far: float = 5.0
    learning_rate: float = 1e-4
    num_epochs: int = 1

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for batch in data_loader:
                B: int = batch.shape[0]
                optimizer.zero_grad()

                positions, _, target_rgb = preprocess_data(batch, num_samples, near, far)

                model_output: torch.Tensor = model.forward(x=positions)

                rgb_pred = model_output[:, :3].reshape(shape=(B, num_samples, 3))
                density = model_output[:, 3].reshape(shape=(B, num_samples, 1)).squeeze(-1)
                depths = positions.reshape(shape=(B, num_samples, 3))[:, :, 2]
                rendered_rgb = volume_rendering(rgb_pred, density, depths)

                loss = nn.MSELoss()(rendered_rgb, target_rgb)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                avg_batch_loss = epoch_loss / (pbar.n + 1)
                pbar.set_postfix(loss=avg_batch_loss)
                pbar.update(1)

        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")