# Team #75: High-Performance Clifford Layers in C

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

## Clifford-ML Layer Benchmarking Pipeline

This repository provides a small “harness+scripts” pipeline to:

1. **Register** multiple implementations of a layer (baseline + optimized variants).  
2. **Benchmark** them for both correctness and performance (cycles & FLOPs).  
3. **Sweep** over batch sizes, emit CSV data, and produce plots.

Three layers are supported:

- **Linear** (Clifford-linear forward)  
- **Activation** (multivector embed-gate-gather)  
- **GroupNorm** (per-group normalization + affine)

---

### Directory layout

```

csrc/
├── linear\_optimization/
│   ├── main.c, linear\_bench.c/.h/.impls/.setup, run\_linear.sh
├── mv\_act\_optimization/
│   ├── main.c, mv\_act\_bench.c/.h/.impls/.setup, run\_act.sh
└── gnn\_optimization/
    ├── main.c, gnn\_bench.c/.h/.impls/.setup, run\_gnn.sh


````

Each layer directory contains:

- `*_bench.[ch]`  — registration + timing harness  
- `*_impls.c`    — baseline + stubbed-out optimized variants  
- `*_setup.[ch]` — randomized input generation & globals  
- `main.c`       — CLI driver (mode 0 vs. mode 1)  
- `run_*.sh`     — compile + run script (sweeps batches, emits CSV, calls plot)

---

### Prerequisites

- **C99** compiler (e.g. `gcc` or `clang`)  
- Linux/macOS (PMU timing via kperf on Apple, or RDTSC on x86)  
- Python 3 + Matplotlib (optional, for plotting)

---

### Build & run

Each layer follows the same pattern:

#### 1. Fixed-input correctness & speed (mode 0)

```bash
cd csrc/linear_optimization
./run_linear.sh 0
````

This will:

* Run `linear_benchmark` for a fixed input
* Print a table of each registered function’s cycles, verify outputs match, and show speedups relative to baseline.

#### 2. Batch-sweep performance logging (mode 1)

```bash
cd csrc/linear_optimization
./run_linear.sh 1
```

That script will:

1. Recompile fresh.
2. Query `./linear_benchmark --count` + `--name` to discover implementations.
3. For each function, for each batch in `(16,32,64,…,16384)`:

   * Run `./linear_benchmark 1 … <batch> <func_idx>`
   * Append `batch,cycles,flops` to `baseline_perfplot_linear.csv`.
4. Call `plot_performance.py baseline_perfplot_linear.csv …` to produce `perf_plot.png`.

The **Activation** and **GroupNorm** directories are identical in structure; just replace `linear` with `activation` or `groupnorm` in commands (`run_act.sh`, `perfplot_activation.csv`, etc.).

---

### CSV format

Each CSV looks like:

```csv
baseline_linear
batch,cycles,flops
16,67133.57,132096
32,132913.65,264192
…

optimized_v1
batch,cycles,flops
16,67050.75,132096
…
```

Plotting scripts expect:

* Blocks separated by a blank line.
* First line of each block: the function name.
* Next line: header `batch,cycles,flops`.

---

### Adding your own layer or variants

1. **Implement** your layer’s `.setup.c/.h` (data generation, globals).
2. **Copy** one of the harnesses (`*_bench.c/.h`, `*_impls.c`) and adjust function pointer types.
3. **Register** your implementations in `register_…_functions()`.
4. **Write** a `run_*.sh` by following the templates.
5. **Run** `./run_*.sh 0` and `./run_*.sh 1`.

---

### Troubleshooting

* **“Abort trap: 6”** often means a null pointer or misuse of the setup globals—verify you called `setup_*()` before any benchmark loops, and `cleanup_*()` afterward.
* If you see **undefined** globals in `*_impls.c`, check that `*_setup.h` declares them as `extern` and that your `*_setup.c` actually defines them.

---


## 📝 To-Dos

### PRESENTATION WEEK 02.05: 
- [ ] All: presentation prep -- SHOULD WRITE SLIDE ABOUT OWN IMPLEMENTATION 
- [ ] Kons : OPEN TO HELP 
- [ ] Kons or Mikke : do final plotting
- [ ] Mikke : clean repo 
- [ ] Mikke : csv on who wrote what (actual implementation as TA asked)
- [ ] Arnie : do roofline plotting and calculations for every layer (say if need to be executed on main computer )
- [ ] Simon : AVX implementation for groupnorm

### WEEK 25.05 - 31.05:
- [ ] Arnie: finish the optimizations for the linear layer (do you consider memory ?)
- [ ] Arnie: have the papi library figured out so that can decide if byte transferred is gonne be analytically done or automatic
- [ ] Mikke, Kons: trying potential optimization for the activation layer 
- [ ] Mikke, Kons: figue out AVX opt
- [ ] Mikke, Kons : figure out to measure the cache misses 
- [ ] Mikke : drafzt of the presentation (based on previous presentations) 
- [ ] Simon : trying potential optimization for the groupnorm layer  (fuse the loops and do computation like transppose in one go)

### WEEK 19.05 - 25.05:
- [X] All : Prepare presentation for meeting 20-30 slides
- [X] Mikke: trying potential optimization for the activation layer 
- [X] Kons, Arnie: trying potential optimization for the linear layer  
- [X] Simon : trying potential optimization for the groupnorm layer  
- [X] Mikke : give extra computer to kons 
- [X] All : banchmark and plot their respective layer opt, + add flop count

### WEEK 12.05 - 18.05:
- [X] Mikke: implement the bias for the activation layer 
- [X] Kons: add the sigmoid to the pipeline 
- [X] Simon: try to include the decomposition 
- [X] Arnie: Implement basic Roofline plot (to improve further)
- [X] Arnie: Improve performance plotting
- [X] Arnie: Implement linear layer optimizations (K-Accumulators, FMA, AVX-2, no flatten/unflatten) (~8.6x max speed-up achieved)
- [X] Simon: trying potential optimization for the batch norm layer 
- [X] Mikke: trying potential optimization for the activation layer 
- FMA (Fused Multiply-Add), Strength reduction (e.g., `* → <<`), Common-sub-expression elimination, Scalar replacement to keep array elements in registers  
- Use loop unrolling and accumulators  
- Leverage SIMD instructions (e.g., SSE, AVX, AVX2, AVX-512)  

### WEEK 05.05 - 11.05:
- [X] All :Make sure the baseline is fully working  
- [X] Mikke : Wrapper files: good – maybe adapt to be able to test new optimizations/versions (maybe future work)  
- [X] Mikke : Test files: Input / output comparison  
- [X] Mikke : For linear layer: 15 tests cases; should maybe have more test cases (groupnorm has 80)  
- [X] For Clifford algebra nonlinearities: Will extend after baseline passes the initial test  
- **Important:** when we have a working version, merge it into `master`  

---

### Layer – Geometric Clifford algebra linear layers (Kons, Arnie)
- [X] FMA (Fused Multiply-Add), Strength reduction (e.g., `* → <<`), Common-sub-expression elimination, Scalar replacement to keep array elements in registers  (ARNIE)
- [X] Use loop unrolling and accumulators (ARNIE)
- [X] Leverage SIMD instructions (AVX2)  (ARNIE)
- [ ] Everywhere that use `malloc` (so sig init and destruct): remove it so it’s not in the performance measure  
- [X] Cache kernel **K** across passes – as we are doing inference (KONS)
- [X] ~~Use BLAS to replace matrix-vector multiplication~~ <- No, we'll need a specialize matrix multiplication fusing the kernels (ARNIE)
- [X] Potentially avoid flattening -> tried this, however seems like bad cache accesses leads to degraded performance (~0.75x baseline) (ARNIE)
- [ ] Implement specialized matrix-matrix multiplication by fusing kernel construction (remove kernel construction & redundant elements, try avoid flattening/unflattening) (ARNIE)
- [ ] Look into Sparse Matrix-Vector multiplication 

---

### Layer – Geometric Clifford algebra nonlinearities (Mikke)
- [X] Fails at certain test – so review the code (MIKKE)  
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
- [X] On the back substitution we could first norm the diagonal since we do backsub many vectors with one mat (numerical stability might arise)  
- [ ] Generally use vectorization (especially if we have the I dimension)  
- [ ] If LAPACKE and BLAS still give issues then include directly in the code base the different requirement – mail the TA to see if he also had this issue  

---

### Benchmarking and plots (Kons, Arnie)
- [ ] With a config/bash file (be able to init some parameters) and then create a CSV and directly plot it  
- [ ] Automate the process  
- [ ] Determine the different perspective to compare:  
  - Py VS C baseline  
  - C baseline VS other optimizations  
- [ ] CYCLE COUNTER: adapted; output format should be automated  
  - For groupnorm: extra init because 2 different timing (because the nice imports that don’t work for everyone 🙂)  
- [ ] PLOTS: write plotting scripts  
- [ ] Performance: Kons already has  
- [X] Roofline: Arnie – take the CSV and do the plotting  
- [ ] Need to consider the bytes transferred -> either do manually or using special PAPI mem-access library
- [ ] Need to decide what’s in cache and what’s in memory ->  -> using special PAPI mem-access library

---

### Project server section update:
- [X] 1. Project description
- [X] 2. Existing implementation
- [X] 3. C-Straightforward
- [X] 4. Validation
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

