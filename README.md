# RigorousRG
This is an implementation of the rigorous RG algorithm first described by Arad, et al., (arXiv:1602.08828) and adapted by Roberts, Vidick, and Motrunich (arXiv:1703.01994).
Based on the ITensor C++ tensor network library by Miles Stoudenmire and Steve White (itensor.org).

To use: first, install ITensor (https://github.com/ITensor/ITensor) version 3.
Then copy the Makefile.example in this repository to Makefile, changing the ITensor installation directory to point to the appropriate location on your machine.
To run RRG for one of the example models, simply use `make` to compile the example.
Then run the compiled executable, providing the example configuration file provided in the appropriate directory as a command-line argument.

Currently the example models which are implemented are a particular nonintegrable Ising model (where the driver code solves for the ground states and bottom of the excitation band), which you can compile with `make rrg_ising`; and the disordered XY model (where the driver code focuses on resolving the gap), which you can compile with `make rrg_random_xy`.
