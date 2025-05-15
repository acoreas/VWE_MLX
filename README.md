# Comparison of MLX and py-metal-compute to solve viscoelastic wave equation (VWE) using FDTD method

---

## Description
This notebook implements a stripped down version of the Viscoelastic Wave Equation (VWE) Finite-Difference Time-Difference (FDTD) calculation found in the [BabelViscoFDTD library](https://github.com/ProteusMRIgHIFU/BabelViscoFDTD). Using both [MLX](https://github.com/ml-explore/mlx) and a [fork of py-metal-compute](https://github.com/ProteusMRIgHIFU/py-metal-compute). Original project is at (py-metal-compute)[https://github.com/baldand/py-metal-compute]. MLX is a high-level library to run functions aimed for ML/AI using Apple Sillicon GPUs. py-metal-compute is aimed mainly for a close-to-hardware interface to run user-defined GPU kernels.

The implementation of the VWE has 3 main kernel functions (stress, particle, and sensors) that are called hundreds of times in a loop as results are updated in small temporal $\delta t$ steps.

## Problem
MLX does not replicate the current metalcompute implementation and the kernel seems to be returning early. BabelViscoFDTD is quite complex and is difficult to isolate only the VWE part of the code. However, We did our best to move the less relevant aspects of the code to the viscoelastic_utils.py file and kept the more important pieces inside a VWE class including kernel setup and execution. To simplify troubleshooting, only the stress kernel is enabled.

During our testing, we implemented the following call in our stress_kernel.c file as an easy check to see how the kernel was behaving. Note EL is a macro to access the Sigma_xy part of one of the output buffers using the i and j indices and is the result that is ultimately plotted at the end.

`EL(Sigma_xy,i,j) += 1; // TEST STATEMENT`

What we noticed is that after a certain point in the kernel, the results are no longer appearing as expected with results defaulting to the mlx init_value and no error being thrown.

## Pre-requisites
Use provided environment.yml file to create conda environment with all dependancies.

MLX version used for testing = 0.22.0
