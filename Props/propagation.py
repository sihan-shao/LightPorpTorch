import sys
sys.path.append('../')
from abc import ABC, abstractmethod
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.nn.functional import pad
from DataType.ElectricField import ElectricField
from utils.Helper_Functions import ft2, ift2
from utils.units import *

class Propagation(ABC, nn.Module):

    @property
    def z(self) -> torch.Tensor:
        return self._z
    
    @z.setter
    def z(self, z : torch.Tensor) -> None:
        self._z = torch.tensor(z, device=self.device)
    
    def compute_padding(self, H, W, padding_scale, return_size_of_padding=False):
        # get the shape of processing
        paddingH = int(np.floor((padding_scale[0] * H) / 2))
        paddingW = int(np.floor((padding_scale[1] * W) / 2))
        paddedH = H + 2*paddingH
        paddedW = W + 2*paddingW
        
        if not return_size_of_padding:
            return paddedH, paddedW
        else:
            return paddingH, paddingW
    
    def create_spatial_grid(self, H, W, dx, dy):
        """
        creates the spatial coordinate grid in x and y direction
        """
        x = torch.linspace(-H * dx / 2, H * dx / 2, H, device=dx.device)
        y = torch.linspace(-W * dx / 2, W * dy / 2, W, device=dy.device)
        meshx, meshy = torch.meshgrid(x, y, indexing='ij')

        return meshx, meshy
    
    def create_frequency_grid(self, H, W, dx, dy):
        """
        creates the frequency coordinate grid in x and y direction
        """
        fx = (torch.linspace(0, H - 1, H, device=dx.device) - (H // 2)) / (H*dx)
        fy = (torch.linspace(0, W - 1, W, device=dy.device) - (W // 2)) / (W*dy)
            
        meshfx, meshfy = torch.meshgrid(fx, fy, indexing='ij')

        return meshfx, meshfy

    @abstractmethod
    def check_crucial_distance(self):
        return NotImplemented
    
    @abstractmethod
    def create_kernel(self):
        return NotImplemented


class Bluestein(ABC):

    def build_CZT_grid(self, 
                       z, 
                       wavelengths,
                       InputHeight, 
                       InputWidth,
                       InputPixel_dx, 
                       InputPixel_dy,
                       outputHeight,
				       outputWidth, 						
				       outputPixel_dx, 
				       outputPixel_dy):
        """
        [From CZT]: Defines the resolution / sampling of initial and output planes.
        
        Parameters:
            z, 
            InputHeight
            InputWidth
            InputPixel_dx,
            InputPixel_dy
            outputHeight
		    outputWidth					
		    outputPixel_dx
		    outputPixel_dy
    
        Returns the set of parameters: 
            Inmeshx: 
            Inmeshy: 
            Outmeshx: 
            Outmeshy: 
            Dm:         dimension of the output ﬁeld
            fy_1:       Starting point along y-direction in frequency range.
            fy_2:       End point along y-direction in frequency range.
            fx_1:       Starting point along x-direction in frequency range.
            fx_2:       End point along x-direction in frequency range.
        """
        # create grid for input plane
        x_in = torch.linspace(-InputHeight * InputPixel_dx / 2, InputHeight * InputPixel_dx / 2, InputHeight, device=self.device)
        y_in = torch.linspace(-InputWidth * InputPixel_dy / 2, InputWidth * InputPixel_dy / 2, InputWidth, device=self.device)
        Inmeshx, Inmeshy = torch.meshgrid(x_in, y_in, indexing='ij')

        # create grid for output plane
        x_out = torch.linspace(-outputHeight * outputPixel_dx / 2, outputHeight * outputPixel_dx / 2, outputHeight, device=self.device)
        y_out = torch.linspace(-outputWidth * outputPixel_dy / 2, outputWidth * outputPixel_dy / 2, outputWidth, device=self.device)
        Outmeshx, Outmeshy = torch.meshgrid(x_out, y_out, indexing='ij')

        wavelengths_expand = wavelengths.view(1, -1, 1, 1)  # reshape to [1, C, :, :] for broadcasting

        # For Bluestein method implementation: 
        # Dimension of the output field - Eq. (11) in [Ref].
        Dm = wavelengths_expand * z / InputPixel_dx

        # (1) for FFT in X-dimension:
        fx_1 = x_out[0] + Dm / 2
        fx_2 = x_out[-1] + Dm / 2
        # (1) for FFT in Y-dimension:
        fy_1 = y_out[0] + Dm / 2
        fy_2 = y_out[-1] + Dm / 2

        return Inmeshx, Inmeshy, Outmeshx, Outmeshy, Dm, fx_1, fx_2, fy_1, fy_2

    def compute_np2(self, x):
        """
        [For Bluestein method]: Exponent of next higher power of 2. 

        Parameters:
        x (float): value

        Returns the exponent for the smallest powers of two that satisfy 2**p >= X for each element in X.
        This function is useful for optimizing FFT operations, which are most efficient when sequence length is an exact power of two.
        """
        return 2**(np.ceil(np.log2(x))).astype(int)

    def compute_fft(self, x, D1, D2, Dm, m, n, mp, M_out, np2):
        """
        [From Bluestein_method]: the FFT part of the algorithm. 
        Parameters:
            x  (float)  : signal
            D1 (float)  : start intermeidate frequency
            D2 (float)  : end intermediate frequency
            Dm (float)  : dimension of the imaging plane.
            m     (int) : original x dimension of signal x
            n     (int) : original y dimension of signal x
            mp    (int) : length of output sequence needed
            M_out (int) : the length of the chirp z-transform of signal x
            np2   (int) : length of the output sequence for efficient FFT computation (exact power of two)

        [Ref] : https://www.osti.gov/servlets/purl/1004350
        """
        # A-Complex exponential term
        A = torch.exp(1j * 2 * torch.pi * D1 / Dm)
        # W-complex exponential term
        W = torch.exp(-1j * 2 * torch.pi * (D2 - D1) / (M_out * Dm))

        # window function (Premultiply data)
        h = torch.arange(-m + 1, max(M_out - 1, m - 1 ) + 1, device=self.device)
        h = W**(h**2 / 2) 
        #print(w.shape)
        h_sliced = h[:mp + 1]
        #print(w_sliced.shape)

        # Compute the 1D Fourier Transform of 1/h up to length 2**nextpow2(mp)
        ft = torch.fft.fft(1 / h_sliced, n=np2, dim=-1) # FFT for Chirp filter [Ref Eq.10 last term]
        #print(ft_w.shape)
        # Compute intermediate result for Bluestein's algorithm [Ref Eq.10 third term]
        b = A**(-(torch.arange(0, m, device=self.device))) * h[..., torch.arange(m - 1, 2 * m - 1, device=self.device)]
        #print(AW.shape)  # torch.Size([1, 28, 1, 200])
        tmp = torch.tile(b, (1, 1, n, 1)).transpose(-2, -1)
        # Compute the 1D Fourier transform of input data
        b = torch.fft.fft(x * tmp, np2, dim=-2)
        # Compute the inverse Fourier transform
        s = torch.tile(ft, (1, 1, n, 1)).transpose(-2, -1)
        b = torch.fft.ifft(b * torch.tile(ft, (1, 1, n, 1)).transpose(-2, -1), dim=-2)
        return b, h

    def Bluestein_method(self, x, f1, f2, Dm, M_out):
        """
        [From CZT]: Performs the DFT using Bluestein method. 
        [Ref1]: Hu, Y., et al. Light Sci Appl 9, 119 (2020).
        [Ref2]: L. Bluestein, IEEE Trans. Au. and Electro., 18(4), 451-455 (1970).
        [Ref3]: L. Rabiner, et. al., IEEE Trans. Au. and Electro., 17(2), 86-92 (1969).
        [Ref4]: https://www.osti.gov/servlets/purl/1004350
    
        Parameters:
            x (jnp.array): Input sequence, x[n] in Eq.(12) in [Ref 1].
            f1 (float): Starting point in frequency range.
            f2 (float): End point in frequency range. 
            Dm (float): Dimension of the imaging plane.
            M_out (float): Length of the transform (resolution of the output plane).
    
        Returns the output X[m].
        """

        # Correspond to the length of the input sequence.  
        _, _, m, n =x.shape
        # intermediate frequency  [1, C, 1, 1]
        D1 = f1 + (M_out * Dm + f2 - f1) / (2 * M_out)   # D1 refer to f1 in [Ref1 Eq.S15]
        # Upper frequency limit
        D2 = f2 + (M_out * Dm + f2 - f1) / (2 * M_out)   # D2 refer to f2 in [Ref1 Eq.S15]

        # Length of the output sequence
        mp = m + M_out - 1
        np2 = self.compute_np2(mp)   # FFT is more efficient when sequence length is an exact power of two.
        b, h = self.compute_fft(x, D1, D2, Dm, m, n, mp, M_out, np2)
        #print(b.shape, w.shape)

        # Extract the relevant portion and multiply by the window function [Ref4 Eq.10 first term]
        b = b[..., m:mp + 1, 0:n].transpose(-2, -1) * torch.tile(h[..., m - 1:mp], (1, 1, n, 1))
        #print(b.shape)
        # create a linearly speed array from 0 to M_out-1
        l = torch.linspace(0, M_out-1, M_out, device=self.device).view(1, 1, 1, -1)
        # scale array to the frequency range [D1, D2]
        l = l / M_out * (D2 - D1) + D1

        # Eq. S14 in Supplementaty Information Section 3 in [Ref1]. Frequency shift to center the spectrum.
        M_shift = -m / 2
        M_shift = torch.tile(torch.exp(-1j * 2 * torch.pi * l * (M_shift + 1 / 2) / Dm), (1, 1, n, 1))
        #print(M_shift)
        # Apply the frequency shift to the final output
        b = b * M_shift
        return b