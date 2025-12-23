# Contributing to GenSBI

Thank you for your interest in contributing to GenSBI! We welcome contributions in the form of bug reports, feature requests, and pull requests.

## Development Setup

To set up your development environment, please follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/aurelio-amerio/GenSBI.git
    cd GenSBI
    ```

2.  **Install dependencies**:
    We recommend using a virtual environment.
    ```bash
    pip install -e ".[examples,validation]"
    ```

## Running Tests

We use `pytest` for testing. To run the test suite:

```bash
pytest test/
```

Ensure all tests pass before submitting a pull request.

## Code Style

Please ensure your code adheres to the existing style conventions. We generally follow PEP 8.

## Adding New Models

If you are adding a new model architecture, please add it to `src/gensbi/models/` and ensure it inherits from the appropriate base classes (e.g., `flax.nnx.Module`). Don't forget to add a corresponding Model Card in the documentation.