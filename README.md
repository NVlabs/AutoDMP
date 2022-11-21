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

Follow the instructions to build DREAMPlace at [README_DREAMPlace.md](README_DREAMPlace.md).

# How to Run Multi-Objective Bayesian Optimization

Run MOBO on NVDLA NanGate45 Bookshelf files, for 20 iterations on 2 GPUs.
```
./tuner/run_tuner.sh 1 1 test/nvdla_nangate45_51/configspace.json test/nvdla_nangate45_51/NV_NVDLA_partition_c.aux test/nvdla_nangate45_51/nvdla_ppa.json \"\" 20 2 0 0 10 test/nvdla_nangate45_51/mobohb_log
```

# Information

The physical design flow in [scripts](scripts) requires the use of netlist/Tcl files from the [TILOS-MacroPlacement](https://github.com/TILOS-AI-Institute/MacroPlacement) repository.
