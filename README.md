# RigorousRG
This is an implementation of the rigorous RG algorithm first described by Arad, et al., (arXiv:1602.08828) and adapted by Roberts, Vidick, and Motrunich (arXiv:1703.01994). Based on the ITensor C++ tensor network library by Miles Stoudenmire and Steve White (itensor.org).

To use: first, install ITensor (https://github.com/ITensor/ITensor) version 3. Then copy the RRG Makefile.example to Makefile, changing the ITensor installation directory to point to the appropriate location on your machine.
To run RRG for one of the example models, use make to compile the example and edit the appropriate configuration file. Then run the compiled executable with the configuration file included in the folder as a command-line argument.

Currently the example models which are implmented are a particular nonintegrable Ising model (which solves for ground- and low-energy states), for which you can use `make rrg_ising`; and the disordered XY model (which focuses on resolving the gap and computes mutual information between the first site and the rest of the chain), for which you can use `make rrg_random_xy`.
