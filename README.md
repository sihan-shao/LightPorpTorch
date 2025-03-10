# ✨ LightPropTorch ✨

**Efficient Multi-Wavelength Propagation Simulation with Vectorization**

**Main Features**

- An abstracted optimization framework based on PyTorch that supports multiple GPUs.

- Custom data types for scalar and vectorial electric fields, which store both spectral and spatial information, along with simple visualization methods.
    - Scalar E-field: \( E(\mathbf{r}) = A(\mathbf{r})e^{-kz} \) is a 4D object with dimensions [1 × C (wavelength) × Height × Width].
    - Vectorial E-field: \( \vec{E} = \left(E_x, E_y, E_z\right) \) is a 4D object with dimensions [3 × C (wavelength) × Height × Width].

- A variety of built-in optical components and propagators for forward modeling.
    - Light sources: Plane wave and Gaussian beam.
    - Optical elements: Various apertures, focusing lenses, and diffractive optical elements.
    - Propagators: Angular Spectrum Method (ASM), Rayleigh-Sommerfeld Convolution (RSC), and Vectorial RSC.

- Parallel simulation for multiple wavelengths using vectorization (broadcasting).

    

