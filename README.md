# Supplementary code for "A tension-induced morphological transition shapes the avian extra-embryonic territory" by Michaut et al.

## Installation

To run the code, please configure a conda environment with the included .yml file. The code uses the FEniCSx open source library for the simulation of bulk-elastic, shear-viscous quail embryo using finite elements. (https://fenicsproject.org). As of October 2024, FEnicSx is still in active development with frequent syntax changes and incompatibility between versions, it is thus recommended to use the pre-configured environment rather than installing a more recent version from scratch.

## Usage

To reproduce the simulations as used in the paper, simply run the notebook. To obtain different results, simply vary the parameters in the first cell. Please note that the code is not production-quality and numerical instabilities can occur if pushed too far beyond the configuration for which it was optimised.
