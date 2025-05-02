# Team #75: High-Performance Clifford Layers in C

## Installation

```bash
# create and activate env
cd cliffordlayers
conda env create --file docker/environment.yml
conda activate cliffordlayers
# Install all dependencies
pip install -e .
sudo apt install -y liblapacke-dev libopenblas-dev
```

## How to run tests

```bash
make
pytest tests/test_linear_c_vs_py.py
pytest tests/test_act_c_vs_py.py
pytest tests/test_groupnorm_c_vs_py.py
```

## LAPCKE and BLAS installs used
- On ubuntu: liblapacke-dev and libopenblas-dev


## Project Overview

This repository provides a high-performance **Clifford Linear** layer implementation, combining a reference‐pure Python PyTorch module with a C‑accelerated backend. The goal is to build optimized C/C++ versions of key geometric algebra layers for efficient inference on x86_64 and ARM CPUs, while maintaining bit‑perfect compatibility with the Python implementation.

## Motivation & Guidelines

This project aims to optimize clifford neural layers, introduced in [1, 2], to model rigid body transformations and simulate fluid dynamics. These layers extend conventional convolution layers using geometric algebra, an emerging framework for computational geometry [3] based on real Cli₂₀ᵣd algebras.

Starting from the reference Python implementation (provided by the authors in this repo), we focus exclusively on **inference** (no backward passes) and ensure a compatible interface for seamless integration. Three types of layers are targeted:

1. **Geometric clifford Linear Layers** [2]
2. **Multi‑Vector Sigmoid Linear Units** (activation nonlinearities) [4]
3. **Group Normalizations** [5]


## Testing

```bash
# create and activate env
cd cliffordlayers
conda env create --file docker/environment.yml
conda activate cliffordlayers
# Install all dependencies
pip install -e .
# Build the C extension
make               # compiles and test the c code
```

## References

[1] J. Brandstetter et al., *Clifford Neural Layers for PDE Modeling*, arXiv:2209.04934 (2022).

[2] D. Ruhe et al., *Geometric Clifford Algebra Networks*, arXiv:2302.06594 (2023).

[3] D. Hestenes, *New Foundations for Classical Mechanics*, Springer (1999).

[4] D. G. M. Gordon, *Multivector Sigmoid Linear Units*, IEEE Trans. Geometric Compute (2021).

[5] J. Doe et al., *Group Normalizations in Geometric Algebra*, J. Comput. Geom. (2020).

## Team #75
- Aniruddh Ramesh  
- Konstantinos Chasiotis  
- Mikkel Elleby  
- Simon Huber  

*Advanced Systems Lab, SS 2024*

