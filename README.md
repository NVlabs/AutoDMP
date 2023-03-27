# AutoDMP: Automated DREAMPlace-based Macro Placement

Built upon the GPU-accelerated global placer [DREAMPlace](https://doi.org/10.1109/TCAD.2020.3003843) and detailed placer [ABCDPlace](https://doi.org/10.1109/TCAD.2020.2971531),
AutoDMP adds simultaneous macro and standard cell placement enhancements and automatic parameter tuning based on multi-objective hyperparameter Bayesian optimization (MOBO).

* Simultaneous Macro and Standard Cell Placement Animations

| MemPool Group | Ariane |
| -------- | ----------- |
| ![MemPool Group](images/mempool.gif) | ![Ariane](images/ariane.gif) |

# Publications

* Anthony Agnesina, Puranjay Rajvanshi, Tian Yang, Geraldo Pradipta, Austin Jiao, Ben Keller, Brucek Khailany, and Haoxing Ren, 
  "**AutoDMP: Automated DREAMPlace-based Macro Placement**", 
  International Symposium on Physical Design (ISPD), Virtual Event, Mar 26-29, 2023 ([preprint](https://research.nvidia.com/publication/2023-03_autodmp-automated-dreamplace-based-macro-placement)) ([blog](https://developer.nvidia.com/blog/autodmp-optimizes-macro-placement-for-chip-design-with-ai-and-gpus/))

# Dependency 

- [DREAMPlace](https://github.com/limbo018/DREAMPlace)
    - Commit b8f87eec1f4ddab3ad50bbd43cc5f4ccb0072892 
    - Other versions may also work, but not tested

- GPU architecture compatibility 6.0 or later (Optional)
    - Code has been tested on GPUs with compute compatibility 8.0 on DGX A100 machine. 

# How to Build 

You can build in two ways:
- Build without Docker by following the instructions of the DREAMPlace build at [README_DREAMPlace.md](README_DREAMPlace.md).
- Use the provided Dockerfile to build an image with the required library dependencies.

# How to Run Multi-Objective Bayesian Optimization

To run the test of multi-objective Bayesian optimization on NVDLA NanGate45, call:
```
./tuner/run_tuner.sh 1 1 test/nvdla_nangate45_51/configspace.json test/nvdla_nangate45_51/NV_NVDLA_partition_c.aux test/nvdla_nangate45_51/nvdla_ppa.json \"\" 20 2 0 0 10 ./tuner test/nvdla_nangate45_51/mobohb_log
```
This will run on the GPUs for 20 iterations with 2 parallel workers. The different settings for the Bayesian optimization can be found in [tuner/run_tuner.sh](tuner/run_tuner.sh). The easiest way to explore different search spaces is to modify [tuner/configspace.json](tuner/configspace.json). You can also run in single-objective mode or modify the parameters of the kernel density estimators in [tuner/tuner_train.py](tuner/tuner_train.py).

# Physical Design Flow

The physical design flow requires RTL, Python, and Tcl files from the [TILOS-MacroPlacement](https://github.com/TILOS-AI-Institute/MacroPlacement) repository. Only the codes that we have added and modified are provided in [scripts](scripts). 
