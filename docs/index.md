# GenSBI 
![Tests](../img/badges/tests.svg)
![Coverage](../img/badges/coverage.svg)

```{image} _static/logo.png
  :alt: GenSBI Logo
  :align: center
  :width: 600px
  :class: logo-transparent-bg
```

## Work in progress!
GenSBI is a work in progress, and we are actively developing new features and improvements. Expect the API and examples to evolve over time. We welcome contributions and feedback from the community!

## Getting Started

To get started with GenSBI, install the package using pip:

```bash
pip install git+https://github.com/aurelio-amerio/GenSBI.git[cuda12]
```

We advise to take a look at the [Getting Started](getting_started.md) page for additional installation instructions and basic usage. 

You can also explore the [Examples](examples.md) page for practical demonstrations of GenSBI's capabilities.

You can find the API documentation in the [API Documentation](api/gensbi/index) section.

## Examples

<img src="https://github.com/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/two_moons/flow_simformer/animated_plot_samples_simformer.gif?raw=true" alt="two-moons posterior sampling" height="400px">
<img src="https://github.com/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/two_moons/flow_simformer/animated_plot_posterior_simformer.gif?raw=true" alt="two-moons posterior sampling" height="400px">

Examples for this library are available separately in the [GenSBI-examples](https://github.com/aurelio-amerio/GenSBI-examples) repository.

Some key examples include:

**Unconditional Density Estimation:**

- `flow_matching_2d_unconditional.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/flow_matching_2d_unconditional.ipynb) <br>
Demonstrates how to use flow matching in 2D for unconditional density estimation.
- `diffusion_2d_unconditional.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/diffusion_2d_unconditional.ipynb) <br>
Demonstrates how to use diffusion models in 2D for unconditional density estimation.

**Conditional Density Estimation:**

- `two_moons_flow_simformer.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/two_moons/flow_simformer/two_moons_flow_simformer.ipynb) <br>
Uses the Simformer model for posterior density estimation on the two-moons benchmark.
- `two_moons_flow_flux.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/two_moons/flow_flux/two_moons_flow_flux.ipynb) <br>
Uses the Flux1 model for posterior density estimation on the two-moons benchmark.
- `gaussian_linear_flow_flux1joint.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/gaussian_linear/flow_flux1joint/gaussian_linear_flow_flux1joint.ipynb) <br>
Uses the Flux1Joint model for posterior density estimation on the Gaussian Linear benchmark.
- `slcp_flow_simformer.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aurelio-amerio/GenSBI-examples/blob/main/examples/sbi-benchmarks/slcp/flow_simformer/slcp_flow_simformer.ipynb) <br>
Uses the Simformer model for posterior density estimation on the SLCP benchmark.    

## Table of Contents
```{toctree}
:maxdepth: 1

Getting Started <getting_started>
Examples <examples>
References <references>
API Documentation <api/gensbi/index>
```


