"""
Common utility functions used across the LightPropTorch library.
This module contains reusable helper functions to reduce code duplication.
"""

import torch
import matplotlib.pyplot as plt
from utils.Visualization_Helper import add_colorbar


def create_circle_mask(input_tensor: torch.Tensor, radius: float = None) -> torch.Tensor:
    """
    Create a circular mask for a 2D tensor.
    
    Args:
        input_tensor (torch.Tensor): Input tensor with shape [H, W]
        radius (float, optional): Radius of the circle. If None, uses min(H, W) / 2
    
    Returns:
        torch.Tensor: Binary mask with 1 inside the circle and 0 outside
    """
    H, W = input_tensor.shape
    # Set default radius if not provided
    if radius is None:
        radius = min(H, W) / 2
    
    # Create a meshgrid
    y, x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
    
    # Compute distance to center
    center_y, center_x = H / 2, W / 2
    dist = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    # Create the mask
    mask = (dist <= radius).float()
    return mask


def visualize_height_map(height_map: torch.Tensor, 
                         circ_aperture: bool = True,
                         cmap: str = 'viridis',
                         figsize: tuple = (4, 4),
                         title: str = '2D Height Map of Hologram'):
    """
    Visualize a height map with optional circular aperture masking.
    
    Args:
        height_map (torch.Tensor): Height map tensor to visualize
        circ_aperture (bool): Whether to apply circular aperture masking
        cmap (str): Matplotlib colormap name
        figsize (tuple): Figure size (width, height)
        title (str): Plot title
    """
    if circ_aperture:
        mask = create_circle_mask(height_map)
        thickness = height_map.detach().cpu().numpy() * mask.detach().cpu().numpy()
    else:
        thickness = height_map.detach().cpu().numpy()
    
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    
    # Create 2D plot
    plt.subplot(1, 1, 1)
    _im1 = plt.imshow(thickness, cmap=cmap)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Show the plots
    add_colorbar(_im1)
    plt.tight_layout()
    plt.show()


def set_default_output_dimensions(input_height: int, 
                                   input_width: int,
                                   input_pixel_dx: torch.Tensor,
                                   input_pixel_dy: torch.Tensor,
                                   output_height: int = None,
                                   output_width: int = None,
                                   output_pixel_dx: torch.Tensor = None,
                                   output_pixel_dy: torch.Tensor = None) -> tuple:
    """
    Set default output dimensions if not provided, using input dimensions.
    
    Args:
        input_height: Input plane height in pixels
        input_width: Input plane width in pixels
        input_pixel_dx: Input pixel spacing in x
        input_pixel_dy: Input pixel spacing in y
        output_height: Output plane height (optional)
        output_width: Output plane width (optional)
        output_pixel_dx: Output pixel spacing in x (optional)
        output_pixel_dy: Output pixel spacing in y (optional)
    
    Returns:
        tuple: (output_height, output_width, output_pixel_dx, output_pixel_dy)
    """
    if output_height is None:
        output_height = input_height
    if output_width is None:
        output_width = input_width
    if output_pixel_dx is None:
        output_pixel_dx = input_pixel_dx
    if output_pixel_dy is None:
        output_pixel_dy = input_pixel_dy
    
    return output_height, output_width, output_pixel_dx, output_pixel_dy


def expand_wavelengths_for_broadcast(wavelengths: torch.Tensor, 
                                      target_dims: int = 4) -> torch.Tensor:
    """
    Expand wavelengths tensor for broadcasting with 4D field data.
    
    Args:
        wavelengths (torch.Tensor): Wavelengths tensor [C] or [1, C, 1, 1]
        target_dims (int): Target number of dimensions (default: 4 for [B, C, H, W])
    
    Returns:
        torch.Tensor: Expanded wavelengths with shape appropriate for broadcasting
    """
    # Ensure wavelengths has one dimension
    wavelengths = wavelengths.view(-1)
    
    # Add dimensions for broadcasting based on target
    if target_dims == 4:
        # Shape: [1, C, 1, 1] for broadcasting with [B, C, H, W]
        return wavelengths.view(1, -1, 1, 1)
    elif target_dims == 3:
        # Shape: [C, 1, 1] for broadcasting with [C, H, W]
        return wavelengths.view(-1, 1, 1)
    else:
        raise ValueError(f"Unsupported target_dims: {target_dims}")


def compute_wavenumber(wavelengths: torch.Tensor) -> torch.Tensor:
    """
    Compute wavenumber (k = 2π/λ) from wavelengths.
    
    Args:
        wavelengths (torch.Tensor): Wavelengths tensor
    
    Returns:
        torch.Tensor: Wavenumber tensor with same shape as input
    """
    return 2 * torch.pi / wavelengths
