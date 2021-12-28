# data_analysis_scripts

This directory contains several pyhton scripts calling the the Metropolis-Monte-Carlo simulation found in [`numerics`](../numerics), and creating plots from the results. 

The hit rate calibraition to obtain the function delta(beta) can be found in [`calibHitRate.py`](calibHitRate.py). The plots for the strong and weak coupling are created by [`calibStrongCoupling.py`](calibStrongCoupling.py) and [`calibWeakCoupling.py`](calibWeakCoupling.py). The phase transition scan can be found in [`partitionEvaluation.py`](partitionEvaluation.py), and the systematic deviation scan in [`partitionEvaluationII.py`](partitionEvaluationII.py). The diagram for the phase scan for different Fibonacci lattices was created by [`fibDetailEval.py`](fibDetailEval.py).

[`executor.py`] handles the calls to the simulation binary found in [`numerics`](../numerics). [`evaluator.R`](evaluator.R) evaluates the results and handles the error estimation implemented by the hadron R package. [`pltLib.py`](pltLib.py) and [`helpers.py`](helpers.py) contain some general functionality for creating the plots.

Most of the data collection scripts contain variables of the type `collectData` or `collectRefData`. For a first execution these need to be set to `True`. Due to prior use this might not always be the case when cloning this repository. They can be used to display and update the diagrams without rerunning the Monte-Carlo Simulation

## scripts in partitions

The [`partitions`](partitions) directory contains some scripts for generating and testing the different discretizations of SU(2). [`fibonacci.py`](partitions/fibonacci.py) generates the fibonacci lattices used together with the `--partition-list` option. The rotation matrices for the 120 cell were found with the help of [`polytopes4d.py`](polytopes4d.py). 

The [`3dGeodesicPics.py`](partitions/3dGeodesicPics.py) file contains the code used to generate the pictures of the icosphere, the volleyball lattice as well as the 3D fibonacci lattice. It makes use of the [`fresnel`](https://github.com/glotzerlab/fresnel) python library. This is a cool raytracing library which was only used, because of it's simplicity. Raytracing for these pictures is however completly overkill and led to silly runtimes.

[`icosaeder.py`](partitions/icosaeder.py) and [`volleyball4d.py`](partitions/volleyball4d.py) contain imlementations of the respective lattices for use with the `--partition-list`. These were used to verify the more advanced implementations. The [`quatternionPlotter.py`](partitions/quatternionPlotter.py) was used to visualize the various lattices.

[`icosphere4d.py`](partitions/icosphere4d.py) contains a failed attempt at creating a higher dimensional version of an icosphere, based on the 600 cell.
