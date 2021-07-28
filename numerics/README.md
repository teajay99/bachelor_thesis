# numerics

This directory contains the implementation of the Metropolis-Monte-Carlo algorithm used for this thesis. It can be build with `make` and run with `./main`. The build requires a CUDA installation, although a CUDA compatible GPU is not required. Building and running was only tested on linux. The following command line options are supported:

```
  -h, --help                  Show Help
      --cuda                  Use NVIDIA GPU for calculation
  -b, --beta arg              Value of Beta for Simulation (default: 2.0)
  -d, --delta arg             Search Radius for Simulation (default: 0.2)
  -l, --lattice-size arg      Size of the lattice (l^4) (default: 8)
      --hits arg              Hits per site per Measurement (default: 10)
  -o, --output arg            Output File Name (default: data.csv)
  -m, --measurements arg      Number of Measurements (default: 1000)
      --multi-sweep arg       Number of Sweeps per Measurement (default: 1)
      --partition-tet         Use the tetrahedral subgroup of SU(2) as a
                              gauge group
      --partition-oct         Use the Octahedral subgroup of SU(2) as a gauge
                              group
      --partition-ico         Use the icosahedral subgroup of SU(2) as a
                              gauge group
      --partition-c5          Use the 5 vertices of the 5-cell as a gauge set
      --partition-c8          Use the 16 vertices of the 8-cell as a gauge
                              set
      --partition-c16         Use the 8 vertices of the 16-cell as a gauge
                              set
      --partition-c24         Use the 24 vertices of the 24-cell as a gauge
                              set (Same as --partition-tet)
      --partition-c120        Use the 600 vertices of the 120-cell as a gauge
                              set
      --partition-c600        Use the 120 vertices of the 600-cell as a gauge
                              set (Same as --partition-ico)
      --partition-list arg    Use custom partition provided by an additional
                              List (.csv) File
      --partition-volley arg  Use volleybal mesh on SU(2)
  -c, --cold                  Cold Start
  -v, --verbose               Verbose output

```

## Source Code

The source Code can be found in the [`src`](src) directory. [`main.cpp`](src/main.cpp) contains the `main` function. It reads in the command line options and then creates and calls the `executor` class accordingly. The `executor` class is defined in [`executor.hpp`](src/executor.hpp) and implemented in [`executor.cu`](src/executor.cu). It's job is to create the initial field configuration and then call the `metropolizer` or `cudaMetropolizer` class depending on wether CUDA is enabled. These are defned and implemented in their respective `.hpp` and `.cu` files, and run the metropolis algorithm and perform the measurements.

The various partitions can be found in the [`gaugeElements`](src/gaugeElements) directory. Calculations involving the lattice action are implemented in [`su2Action.hpp`](src/su2Action.hpp). Loading the element list for the `--partition-list` option is done by [`discretizer.hpp`](src/discretizer.hpp) class.
