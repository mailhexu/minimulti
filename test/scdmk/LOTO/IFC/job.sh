#!/bin/bash
cd /home/hexu/projects/minimulti/test/scdmk/LOTO/IFC
# OpenMp Environment
export OMP_NUM_THREADS=1
# Commands before execution
export PATH=/home/hexu/projects/abinit/build_gfortran:$PATH

mpirun  -n 1 anaddb < /home/hexu/projects/minimulti/test/scdmk/LOTO/IFC/run.files > /home/hexu/projects/minimulti/test/scdmk/LOTO/IFC/run.log 2> /home/hexu/projects/minimulti/test/scdmk/LOTO/IFC/run.err
