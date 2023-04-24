#!/bin/bash
cd /home/hexu/projects/minimulti/test/scdmk/PbTiO3/ifc
# OpenMp Environment
export OMP_NUM_THREADS=1
# Commands before execution
export PATH=/home/hexu/projects/abinit/build_gfortran:$PATH

mpirun  -n 1 anaddb < /home/hexu/projects/minimulti/test/scdmk/PbTiO3/ifc/run.files > /home/hexu/projects/minimulti/test/scdmk/PbTiO3/ifc/run.log 2> /home/hexu/projects/minimulti/test/scdmk/PbTiO3/ifc/run.err
