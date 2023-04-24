

## Lattice Wannier Function 

Here are the energy dispersion curve (I didn't use phonon frequency for simplicity) of lattice wannier functions using the SCDM method.  The results are obtained by using the $\Gamma$ modes as anchor points. Each one consists of three bands degenerate at $\Gamma$. 

## Disentanglement based on projection only

I do not use energy window here, therefore all the modes are considered equal. It is quite clear that at many points, the effective phonon frequency is not a selected value from the original phonon, but a averaged one.  

### BaTiO3

- branch 1

![BaTiO3_branch1](/home/hexu/projects/minimulti/test/scdmk/BaTiO3_branch1.png)

* branch 2

![BaTiO3_branch2](/home/hexu/projects/minimulti/test/scdmk/BaTiO3_branch2.png)

* branch 3

![BaTiO3_branch3](/home/hexu/projects/minimulti/test/scdmk/BaTiO3_branch3.png)

* branch 4

![BaTiO3_branch4](/home/hexu/projects/minimulti/test/scdmk/BaTiO3_branch4.png)

* branch 5

![BaTiO3_branch5](/home/hexu/projects/minimulti/test/scdmk/BaTiO3_branch5.png)





### PbTiO3

* branch 1

![PTO_branch_0](/home/hexu/projects/minimulti/test/scdmk/PbTiO3/PTO_branch_0.png)

* branch 2

![PTO_branch_0](/home/hexu/projects/minimulti/test/scdmk/PbTiO3/PTO_branch_1.png)

* branch 3

![PTO_branch_0](/home/hexu/projects/minimulti/test/scdmk/PbTiO3/PTO_branch_2.png)

* branch 4

![PTO_branch_0](/home/hexu/projects/minimulti/test/scdmk/PbTiO3/PTO_branch_3.png)

* branch 5

![PTO_branch_0](/home/hexu/projects/minimulti/test/scdmk/PbTiO3/PTO_branch_4.png)


## Disentanglement using energy window (+ projection)

Now we come back to the first branch of BaTiO3, but add a energy window to it.

The energy window is realized with a fermi function, so below Fermi energy, the weight of bands are about 1, and above Fermi energy are about 0 (which means they are not mixed into the effective wavefunctions.) I use a large smearing of 1eV to keep the bands smooth.

We can see by using a small energy window the energy is more close to the original one but less smooth.  So if we want a good description only at low energy, the energy window scheme might still be useful.

* Efermi= 0

  ![BTO_branch_0_emax_0](/home/hexu/projects/minimulti/test/scdmk/BTO_branch_0_emax_0.png)

* Efermi= 3

* ![BTO_branch_0_emax_3](/home/hexu/projects/minimulti/test/scdmk/BTO_branch_0_emax_3.png)

*  Efermi = 5



![BTO_branch_0_emax_5](/home/hexu/projects/minimulti/test/scdmk/BTO_branch_0_emax_5.png)