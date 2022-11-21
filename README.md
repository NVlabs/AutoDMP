# AutoDMP

Automated DREAMPlace-based Macro Placement (AutoDMP).

Built upon the GPU-accelerated global placer *DREAMPlace* and detailed placer *ABCDPlace*,
AutoDMP adds concurrent macro and standard cell placement (CMP) enhancements and automatic parameter tuning based on multi-objective hyperparameter Bayesian optimization (MOBO).

# Dependency 

- [DREAMPlace](https://github.com/limbo018/DREAMPlace)
    - Commit b8f87eec1f4ddab3ad50bbd43cc5f4ccb0072892 
    - Other versions may also work, but not tested

- GPU architecture compatibility 6.0 or later (Optional)
    - Code has been tested on GPUs with compute compatibility 8.0 on DGX A100 machine. 

# How to Build 

You can build two ways:
- Build without using Docker by following the instructions of the DREAMPlace build at [README_DREAMPlace.md](README_DREAMPlace.md).
- Use the provided Dockerfile to build an image with the required library dependencies.

# How to Run Multi-Objective Bayesian Optimization

To run the test of multi-objective Bayesian optimization on NVDLA NanGate45, simply run:
```
./tuner/run_tuner.sh 1 1 test/nvdla_nangate45_51/configspace.json test/nvdla_nangate45_51/NV_NVDLA_partition_c.aux test/nvdla_nangate45_51/nvdla_ppa.json \"\" 20 2 0 0 10 ./tuner test/nvdla_nangate45_51/mobohb_log
```
This will run on the GPUs for 20 iterations with 2 parallel workers. The different settings for the Bayesian optimization can be found in [tuner/run_tuner.sh](tuner/run_tuner.sh). The easiest way to explore different search spaces is to modify [tuner/configspace.json](tuner/configspace.json). You can also run in single-objective mode or modify the parameters of the kernel density estimators in [tuner/tuner_train.py](tuner/tuner_train.py).

# Physical design flow

The physical design flow requires the RTL, Python, and Tcl files from the [TILOS-MacroPlacement](https://github.com/TILOS-AI-Institute/MacroPlacement) repository. Only the codes that we have added and modified are provided in [scripts](scripts). 
