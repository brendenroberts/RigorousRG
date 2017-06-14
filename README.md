# RigorousRG
This is an implementation of the rigorous RG algorithm first described by Arad, et al., (arXiv:1602.08828) and adapted by Roberts, Vidick, and Motrunich (arXiv:1703.01994). Based on ITensor tensor network library by Miles Stoudenmire and Steve White (itensor.org).

To use: first, install ITensor (https://github.com/ITensor/ITensor). Then, enter one of the directories and copy Makefile.example to Makefile, changing the ITensor directory to point to the appropriate location on your machine. Then make and type rrg to see the correct order of command line parameters. Further paramertes may be changed in the .cpp source code in each directory, including Hamiltonian details and computational settings, and the global maximum truncation error eps is set in rrg.h.

Currently a particular nonintegrable Ising model (RRG) and the random isotropic XY (RXY) model are implemented.
