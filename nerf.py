import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class NeRFModel(nn.Module):
    """
    Neural Radiance Fields (NeRF) Model.

    Args:
        D (int): Number of layers in the MLP.
        W (int): Width of each layer.
        input_ch (int): Number of input channels for the position vector.
        input_ch_dir (int): Number of input channels for the direction vector.
        output_ch (int): Number of output channels (RGB + density).
    """
    def __init__(self, D=8, W=256, input_ch=3, input_ch_dir=3, output_ch=4):
        super(NeRFModel, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_dir = input_ch_dir
        self.output_ch = output_ch

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if (i != 4) else nn.Linear(W + input_ch, W) for i in range(1,D)]
        )
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_dir + W, W // 2)])

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, x):
        """
        Forward pass of the NeRF model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_ch + input_ch_dir).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_ch).
        """
        input_pts, input_dirs = torch.split(x, [self.input_ch, self.input_ch_dir], dim=-1)
        print(f"input_pts shape: {input_pts.shape}")
        print(f"input_dirs shape: {input_dirs.shape}")
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            # Skip connection: concatenate input_pts with h every 4 layers
            if i == 4:
                h = torch.cat([input_pts, h], -1)
                print(f"h shape after skip connection at layer {i}: {h.shape}")

            h = nn.functional.relu(l(h))
            print(f"h shape after layer {i}: {h.shape}")

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_dirs], -1)
        print(f"h shape after concatenating feature and input_dirs: {h.shape}")

        for l in self.views_linears:
            h = nn.functional.relu(l(h))

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], -1)
        return outputs

def sample_along_rays(origins, directions, num_samples, near, far):
    """
    Samples points along rays.

    Args:
        origins (torch.Tensor): Tensor of shape (B, 3) representing the origins of the rays.
        directions (torch.Tensor): Tensor of shape (B, 3) representing the directions of the rays.
        num_samples (int): Number of samples to take along each ray.
        near (float): Near bound for sampling.
        far (float): Far bound for sampling.

    Returns:
        torch.Tensor: Tensor of shape (B, num_samples, 3) representing the sampled points along the rays.
    """
    t_vals = torch.linspace(near, far, num_samples)
    sample_points = origins[:, None, :] + directions[:, None, :] * t_vals[None, :, None]
    return sample_points

def preprocess_data(data : torch.Tensor, num_samples : int, near : float, far : float):
    """
    Preprocesses the input data for the NeRF model.

    Args:
        data (torch.Tensor): Tensor of shape (B, 9) where:
            [:, :3] is position,
            [:, 3:6] is direction, 
            [:, 6:] is target RGB.
        num_samples (int): Number of samples to take along each ray.
        near (float): Near bound for sampling.
        far (float): Far bound for sampling.

    Returns:
        tuple: A tuple containing:
            positions (torch.Tensor): Tensor of shape (B * num_samples, 3) -> sampled points along the rays.
            
            directions_expanded (torch.Tensor): Tensor of shape (B * num_samples, 3) -> expanded directions.
            
            target_rgb (torch.Tensor): Tensor of shape (B, 3) -> target RGB values.
    """
    B : int = data.shape[0]
    origins : torch.Tensor = data[:, :3]
    directions : torch.Tensor = data[:, 3:6]
    target_rgb : torch.Tensor = data[:, 6:]
    
    positions = sample_along_rays(origins, directions, num_samples, near, far)
    
    return (positions.reshape(shape=(B * num_samples, 3)), 
            directions.repeat_interleave(repeats=num_samples, dim=0), 
            target_rgb)

def volume_rendering(rgb, density, depths):
    # rgb: (B, num_samples, 3)
    # density: (B, num_samples, 1)
    # depths: (B, num_samples)
    # Returns: (B, 3)
    delta = torch.cat([depths[:, 1:] - depths[:, :-1], torch.tensor([1e10]).expand(depths[:, :1].shape)], -1)
    alpha = 1 - torch.exp(-density * delta)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1 - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_final = torch.sum(weights[:, :, None] * rgb, -2)
    return rgb_final

@torch.no_grad()
def infer(model: NeRFModel, position: torch.Tensor, direction: torch.Tensor):
    pass

@torch.enable_grad()
def train(model: NeRFModel, data_loader: DataLoader):
    num_samples = 64
    near = 1.0
    far = 5.0
    learning_rate = 1e-4
    num_epochs = 100

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch in data_loader:  
            B = batch.shape
            optimizer.zero_grad()
            
            positions, directions, target_rgb = preprocess_data(batch, num_samples, near, far)
            
            model_output = model(torch.cat([positions, directions], dim=-1))
            rgb_output = model_output[:, :3].reshape(B, num_samples, 3)
            density_output = model_output[:, 3].reshape(B, num_samples, 1)
            
            depths = positions.reshape(B, num_samples, 3)[:, :, 2]  
            rendered_rgb = volume_rendering(rgb_output, density_output, depths)
            
            loss = nn.MSELoss()(rendered_rgb, target_rgb)
            
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")