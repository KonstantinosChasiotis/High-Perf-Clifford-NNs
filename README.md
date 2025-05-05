# Team #75: High-Performance Clifford Layers in C

## 📝 To-Dos

### For this week:
- [ ] Make sure the baseline is fully working  
- [ ] Wrapper files: good – maybe adapt to be able to test new optimizations/versions (maybe future work)  
- [ ] Test files: Input / output comparison  
- [ ] For linear layer: 15 tests cases; should maybe have more test cases (groupnorm has 80)  
- [ ] For Clifford algebra nonlinearities: Will extend after baseline passes the initial test  
- [ ] For groupnorm: others should review, but for now test every feature (just not the rolling mean)  
- [ ] **Important:** when we have a working version, merge it into `master`  

---

### Layer – Geometric Clifford algebra linear layers (Kons, Mikke)
- [ ] FMA (Fused Multiply-Add), Strength reduction (e.g., `* → <<`), Common-sub-expression elimination, Scalar replacement to keep array elements in registers  
- [ ] Use loop unrolling and accumulators  
- [ ] Leverage SIMD instructions (e.g., SSE, AVX, AVX2, AVX-512)  
- [ ] Everywhere that use `malloc` (so sig init and destruct): remove it so it’s not in the performance measure  
- [ ] Cache kernel **K** across passes – as we are doing inference  
- [ ] Use BLAS to replace matrix-vector multiplication  
- [ ] Potentially avoid flattening  
- [ ] Look into Sparse Matrix-Vector multiplication  

---

### Layer – Geometric Clifford algebra nonlinearities (Mikke, Arnie)
- [ ] Fails at certain test – so review the code (MIKKE)  
- [ ] FMA (Fused Multiply-Add), Strength reduction (e.g., `* → <<`), Common-sub-expression elimination, Scalar replacement to keep array elements in registers  
- [ ] Use loop unrolling and accumulators  
- [ ] Leverage SIMD instructions (e.g., SSE, AVX, AVX2, AVX-512)  
- [ ] Using k-accumulators for the convolution instead of `sum +=` which causes read dependencies  
- [ ] Try using blocking in case it improves cache-locality depending on channel/batch size  
- [ ] Using vectorization will certainly help  

---

### Layer – Group normalizations (Simon, Kons)
- [ ] FMA (Fused Multiply-Add), Strength reduction (e.g., `* → <<`), Common-sub-expression elimination, Scalar replacement to keep array elements in registers  
- [ ] Use loop unrolling and accumulators  
- [ ] Leverage SIMD instructions (e.g., SSE, AVX, AVX2, AVX-512)  
- [ ] Don’t permute  
- [ ] Note the expensive things often have I dimensions; the only I dimension we need to support are 2, 4 and 8 – maybe we could create super-optimized version for these input sizes  
- [ ] On the back substitution we could first norm the diagonal since we do backsub many vectors with one mat (numerical stability might arise)  
- [ ] Generally use vectorization (especially if we have the I dimension)  
- [ ] If LAPACKE and BLAS still give issues then include directly in the code base the different requirement – mail the TA to see if he also had this issue  

---

### Benchmarking and plots (Kons, Arnie)
- [ ] With a config/bash file (be able to init some parameters) and then create a CSV and directly plot it  
- [ ] Automate the process  
- [ ] Determine the different perspective to compare:  
  - Py VS C baseline  
  - C baseline VS other optimizations  
- [ ] FLOPS: See implementation with macros to count the flops and COMPARE it with the current counters (Mikke needs to push the `instruction.h` file)  
- [ ] CYCLE COUNTER: adapted; output format should be automated  
  - For groupnorm: extra init because 2 different timing (because the nice imports that don’t work for everyone 🙂)  
- [ ] PLOTS: write plotting scripts  
- [ ] Performance: Kons already has  
- [ ] Roffline: Arnie – take the CSV and do the plotting  
- [ ] Need to consider the bytes transferred ?? for now assume some data is there  
- [ ] Need to decide what’s in cache and what’s in memory  

---

### Project server section update:
- [ ] 1. Project description
- [ ] 2. Existing implementation
- [ ] 3. C-Straightforward
- [ ] 4. Validation
- [ ] 5. Timing
- [ ] 6. Cost analysis
- [ ] 7. Performance plot
- [ ] 8. Benchmark alternatives
- [ ] 9. Bottlenecks
- [ ] 10. Std. C Optimizations
- [ ] 11. Vectorize AVX
- [ ] 12. Plot every improvement
- [ ] 13. Additional analysis
- [ ] 14. Compilers & flags/Processors
- [ ] 15. Roofline

---

### Final presentation and paper:
- [ ] Draft and proofread final presentation  
- [ ] Write project paper  


## Additional infos : 


### How to structure the optimizations:
- Have it written like in the homeworks:  
  - Top of the files are the configs, macros, kernel computation, set notations – maybe in a separate file  
  - Then have separate functions for the baseline and each optimizations  


### CPU INFOS (of the testing laptop)
| Parameter                                                                                   | Value                                             |
|---------------------------------------------------------------------------------------------|---------------------------------------------------|
| Processor manufacturer, name, number, microarchitecture                                     | Intel Core i5-1035G4, Ice Lake                    |
| CPU Base Frequency                                                                          | 1.10 GHz (nominal)                                |
| CPU Maximum Frequency, Frequency Scaling support                                            | 3.70 GHz, supports Turbo Boost                    |
| Phase in Intel’s development model                                                          | Tick (Ice Lake is a process shrink)               |



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


### Project Overview

This repository provides a high-performance **Clifford Linear** layer implementation, combining a reference‐pure Python PyTorch module with a C‑accelerated backend. The goal is to build optimized C/C++ versions of key geometric algebra layers for efficient inference on x86_64 and ARM CPUs, while maintaining bit‑perfect compatibility with the Python implementation.

## Motivation & Guidelines

This project aims to optimize clifford neural layers, introduced in [1, 2], to model rigid body transformations and simulate fluid dynamics. These layers extend conventional convolution layers using geometric algebra, an emerging framework for computational geometry [3] based on real Cli₂₀ᵣd algebras.

Starting from the reference Python implementation (provided by the authors in this repo), we focus exclusively on **inference** (no backward passes) and ensure a compatible interface for seamless integration. Three types of layers are targeted:

1. **Geometric clifford Linear Layers** [2]
2. **Multi‑Vector Sigmoid Linear Units** (activation nonlinearities) [4]
3. **Group Normalizations** [5]


### References

[1] J. Brandstetter et al., *Clifford Neural Layers for PDE Modeling*, arXiv:2209.04934 (2022).

[2] D. Ruhe et al., *Geometric Clifford Algebra Networks*, arXiv:2302.06594 (2023).

[3] D. Hestenes, *New Foundations for Classical Mechanics*, Springer (1999).

[4] D. G. M. Gordon, *Multivector Sigmoid Linear Units*, IEEE Trans. Geometric Compute (2021).

[5] J. Doe et al., *Group Normalizations in Geometric Algebra*, J. Comput. Geom. (2020).

### Team #75
- Aniruddh Ramesh  
- Konstantinos Chasiotis  
- Mikkel Elleby  
- Simon Huber  

*Advanced Systems Lab, SS 2024*

