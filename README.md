# RigorousRG
This is an implementation of the rigorous RG algorithm first described by Arad, et al., (arXiv:1602.08828) and adapted by Roberts, Vidick, and Motrunich (arXiv:1703.01994). Based on ITensor tensor network library by Miles Stoudenmire and Steve White (itensor.org).

To use: first, install ITensor (https://github.com/ITensor/ITensor). Then, enter one of the RRG directories which implement different systems and copy Makefile.example to Makefile, changing the ITensor and ARPACK++ directories to point to the appropriate locations on your machine. RRG can be run without ARPACK++, in which case the iterative solver defined in davidson.cpp is used instead, but this may substantially increase your runtimes. Make and run rrg without command-line arguments in order to see correct usage. Further parameters may be changed in the .cpp source code in each directory, including Hamiltonian details and computational settings, and the global maximum truncation errors eps and epx are set in rrg.h and should be changed depending on which model is being computed.

Currently a particular nonintegrable Ising model and the disordered Z2-symmetric XYZ model are implemented.
