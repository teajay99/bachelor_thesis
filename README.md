# Discretizations approaches for SU(2) and their effects on lattice gauge theories

Hello there,

my name is Timo Jakobs and this repository contains my bachelor's thesis in physics supervised by Prof. Dr. Carsten Urbach from the university of Bonn. The thesis itself can be found [here](https://raw.githubusercontent.com/teajay99/bachelor_thesis/main/thesis/thesis.pdf). The three directories found here contain the following things:

## numerics

The [numerics](numerics) directory contains the implementation of the Metropolis-Monte-Carlo algorithm used for this thesis. It uses C++ and makes use of Nvidia's CUDA library for parallel execution on GPUs. Further details can be found [here](numerics/README.md).

## data_analysis_scripts

The [data_analysis_scripts](data_analysis_scripts) directory contains several python scripts calling the simulation found in the (numerics)[numerics] folder, and creates the diagrams found in the thesis from the results. Further details can be found [here](data_analysis_scrpits/README.md). 

## thesis

In the [thesis](thesis) folder, the LaTeX source code of the final report can be found. It also contain some symbolic links to plots and pictures created within the export folder in the [data_analysis_scripts](data_analysis_scripts) folder. These are unresolved, when initially cloning the repository, as the files are created by the python scripts found in the [data_analysis_scripts](data_analysis_scripts).
