# ✨ LightPropTorch ✨

**Efficient Multi-Wavelength Propagation Simulation**

**Main Features**

- An abstracted optimization framework based on Pure PyTorch that supports multiple GPUs.

- Custom data types for scalar and vectorial electric fields, which store both spectral and spatial information, along with simple visualization methods.
    - Scalar E-field: $E(\mathbf{r}) = A(\mathbf{r})e^{-kz}$ is a 4D object with dimensions [1 × C (wavelength) × Height × Width].
    - Vectorial E-field: $\vec{E} = (E_x, E_y, E_z)$ is a 4D object with dimensions [3 × C (wavelength) × Height × Width].

- A variety of built-in optical components and propagators for forward modeling.
    - Light sources: Plane wave and Gaussian beam.
    - Optical elements: Various apertures, focusing lenses, and diffractive optical elements.
    - Propagators: Angular Spectrum Method (ASM), Rayleigh-Sommerfeld Convolution (RSC), and Vectorial RSC.

- Parallel simulation for multiple wavelengths using broadcasting.

- automatic critical distance calculation to ensure the simulation accuracy.
    
## Installation

### Prerequisite
Our code uses Pytorch 2.1.0 or higher which can be accelerated by multiple GPUs.

Running some of the scripts may require installing additional Python packages. Just follow the terminal hints, for example install the following:

```shell
pip install imageio opencv-python scikit-image
```

## Getting Start and Examples


## Related Projects

Below are some projects developed using this codebase.

- **Neural Dispersive Hologram for Computational Submillimeter-wave Imaging (Master's Thesis):** [Link](https://version.aalto.fi/gitlab/shaos3/NeuralDispersiveHologram)

- **Quantized THz Diffractive Optics Design via Automatic Differentiation:** [Link](https://version.aalto.fi/gitlab/shaos3/ad-thz-diffractiveoptics)

- **Image Deblurring with Neural Networks Using Fourier Optics:** [Link](https://github.com/sihan-shao/DeblurNN/tree/master)