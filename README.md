# thrust_random_walk

Random walk implementation on CUDA-Thrust.

## Overview

This project implements a random walk simulation using CUDA and Thrust libraries. The simulation leverages the parallel processing capabilities of CUDA to efficiently compute the random walk steps.

## Prerequisites

- CUDA Toolkit
- Thrust Library
- CMake
- ParaView (for visualization)

## Building the Project

To build the project, follow these steps:

1. Clone the repository:

   ```sh
   git clone <repository-url>
   cd thrust_random_walk
   ```

2. Create a build directory and navigate into it:

   ```sh
   mkdir build
   cd build
   ```

3. Run CMake to configure the project:

   ```sh
   cmake ..
   ```

4. Build the project using Make:
   ```sh
   make
   ```

## Running the Simulation

After building the project, you can run the simulation with the following command:

```sh
./thrust_random_walk
```

The output files will be generated in the output directory.

## Visualization in ParaView

To visualize the simulation results in ParaView, follow these steps:

1. Open ParaView.
2. Navigate to File -> Open and select the .vtp files from the output directory.
3. Click Apply to load the data.
4. Use the visualization tools in ParaView to explore the random walk simulation.

![Random Walk Visualization](result.avi)
