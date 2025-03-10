# ✨ LightPropTorch ✨

**Efficient Multi-Wavelength Propagation Simulation with Vectorization**

**Main Features**

- An abstracted optimization framework based on PyTorch that supports multiple GPUs.

- Custom data types for scalar and vectorial electric fields, which store both spectral and spatial information, along with simple visualization methods.
    - Scalar E-field: 𝐸(𝑟)=𝐴(𝑟)𝑒^(−𝑘𝑧) is a 4D object with dimensions [1 × C (wavelength) × Height × Width].
    - Vectorial E-field: 𝐸⃗ = (𝐸ₓ, 𝐸ᵧ, 𝐸𝑧) is a 4D object with dimensions [3 × C (wavelength) × Height × Width].

- A variety of built-in optical components and propagators for forward modeling.
    - Light sources: Plane wave and Gaussian beam.
    - Optical elements: Various apertures, focusing lenses, and diffractive optical elements.
    - Propagators: Angular Spectrum Method (ASM), Rayleigh-Sommerfeld Convolution (RSC), and Vectorial RSC.

- Parallel simulation for multiple wavelengths using vectorization (broadcasting).

    

