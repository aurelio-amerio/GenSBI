# Installation


To avoid dependency issues, it is recommended to create a new conda/mamba environment.

```bash
conda create -n gensbi python=3.12 -y
conda activate gensbi
```

To install, clone the repository and install dependencies:

```bash
pip install git+https://github.com/aurelio-amerio/GenSBI.git
```

If a GPU is available, it is advisable to install the cuda version of the package:

```bash
pip install "gensbi[cuda12] @ git+https://github.com/aurelio-amerio/GenSBI.git"
```

Although not mandatory, it is recommended to install also the optional validation package, which provides utilities for evaluating inference performance:

```bash
pip install "gensbi[validation] @ git+https://github.com/aurelio-amerio/GenSBI.git" --extra-index-url https://download.pytorch.org/whl/cpu
```

If you want to run the examples, install the GenSBI-examples repository:

```bash
pip install "gensbi[examples] @ git+https://github.com/aurelio-amerio/GenSBI.git" 
```

To install all the optional dependencies at once, run:

```bashts
pip install "gensbi[cuda12,examples,validation] @ git+https://github.com/aurelio-amerio/GenSBI.git" --extra-index-url https://download.pytorch.org/whl/cpu
```

## Requirements

- Python 3.11+
- JAX
- Flax
- (See `pyproject.toml` for full requirements)