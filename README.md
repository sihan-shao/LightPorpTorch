# ✨ LightPropTorch ✨

**Efficient Multi-Wavelength Propagation Simulation Framework**

**Main Features**

- An abstracted optimization framework implemented by pure PyTorch that supports multiple GPUs.

- Custom data types for scalar and vectorial electric fields, which store both spectral and spatial information, along with simple visualization methods.
    - Scalar E-field: $E(\mathbf{r}) = A(\mathbf{r})e^{-kz}$ is a 4D object with dimensions [1 × C (wavelength) × Height × Width].
    - Vectorial E-field: $\vec{E} = (E_x, E_y, E_z)$ is a 4D object with dimensions [3 × C (wavelength) × Height × Width].

- A variety of built-in optical components and propagators for forward modeling.
    - Light sources: Plane wave and Gaussian beam.
    - Optical elements: Various apertures, focusing lenses, and diffractive optical elements.
    - Propagators: Angular Spectrum Method (ASM), Rayleigh-Sommerfeld Convolution (RSC), Vectorial RSC, Fresnel and Fraunhofer propagators.

- Parallel simulation for multiple wavelengths using broadcasting.

- Automatic critical distance calculation to ensure the simulation accuracy.
    
## Installation

### Prerequisite
Our code uses Pytorch 2.1.0 or higher which can be accelerated by multiple GPUs.

Running some of the scripts may require installing additional Python packages. Just follow the terminal hints, for example install the following:

```shell
pip install imageio opencv-python scikit-image
```

## TODO List

- [ ] Make the sampling size of the Fraunhofer Propagator at the target plane the same for various wavelength.

- [ ] A 4-f system example.

- [ ] Seperate SLM and DOE class. 

- [ ] Polarization modulation SLM for vectorial E-field (adds phase mask to $E_x$ and $E_y$ independently).

- [ ] More Polarization-based elements such as beam splitter and linear polarizer. 

- [ ] Chirped z-transform propagation algorithm. Ref: "Efficient full-path optical calculation of scalar and vector diffraction using the Bluestein method"

- [ ] Off-axis propagation algorithm. Ref: "Modeling off-axis diffraction with least-sampling angular spectrum method."


## Getting Start and Examples

- **Basics** 

    [1-basics.ipynb](./examples/1-basics.ipynb) introduces how to build a basic datatype for wave propagation simulation and cases using ASM and RSC method.

- **Vectorial Field Propagation and Visualization**

    [2-vectorial_prop_vis.ipynb](./examples/2-vectorial_prop_vis.ipynb) demonstrates how a customized vectorial Gaussian beam propagates and shows the visualization of the $E_x$, $E_y$, and $E_z$ components and polarization analysis.

- **Diffractive Optical Element Design via Automatic Differentiation**

    [3-DOEdesign.ipynb](./examples/3-DOEdesign.ipynb.ipynb) demonstrates how to optimize a full-precision and quantization diffractive optical element to generate a holographic images.

## Related Projects

Below are some projects developed using this codebase.

- **Neural Dispersive Hologram for Computational Submillimeter-wave Imaging (Master's Thesis):** [Link](https://version.aalto.fi/gitlab/shaos3/NeuralDispersiveHologram)

- **Quantized THz Diffractive Optics Design via Automatic Differentiation:** [Link](https://version.aalto.fi/gitlab/shaos3/ad-thz-diffractiveoptics)

- **Image Deblurring with Neural Networks Using Fourier Optics:** [Link](https://github.com/sihan-shao/DeblurNN/tree/master)