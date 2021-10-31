import numpy as np
from netCDF4 import Dataset
from ase import Atoms


def save_ifc_to_netcdf(fname, ifc, Rlist, atoms, ref_energy=0.0):
    """
    save IFC to netcdf.
    """
    nR = len(Rlist)
    ifc_vals = ifc
    natom = len(atoms)
    root = Dataset(fname, 'w')
    id_three = root.createDimension("three", 3)
    id_natom = root.createDimension("natom", natom)
    id_natom3 = root.createDimension("natom3", natom * 3)
    id_ifc_nR = root.createDimension("ifc_nR", nR)

    ref_masses = root.createVariable("ref_masses", 'f8', ("natom", ))
    ref_masses.description = "REFerence atom MASSES"
    ref_zion = root.createVariable("ref_zion", 'i4', ("natom", ))
    ref_zion.description = "REFerence atom ZION"
    ref_cell = root.createVariable("ref_cell", 'f8', ("three", "three"))

    ref_cell.description = "REFerence structure CELL"
    ref_xred = root.createVariable("ref_xred", 'f8', ("natom", "three"))

    ref_xred.description = "REFerence structure XRED"

    ref_xcart = root.createVariable("ref_xcart", 'f8', ("natom", "three"))

    ref_xred.description = "REFerence structure XCART"



    ifc_Rlist = root.createVariable("ifc_Rlist", "i4", ("ifc_nR", "three"))

    ifc_Rlist.description = "IFC RLIST"
    ifc_vallist = root.createVariable("ifc_vallist", "f8",
                                      ("ifc_nR", "natom3", "natom3"))
    ifc_vallist_imag = root.createVariable("ifc_vallist_imag", "f8",
                                      ("ifc_nR", "natom3", "natom3"))


    ifc_ref_energy = root.createVariable("ref_energy", "f8", ())

    ifc_vallist.description = "IFC VALUE LIST"
    ref_cell.unit = "Angstrom"
    ref_xcart.unit = "Angstrom"
    ref_masses.unit = "atomic"
    ifc_vallist.unit = "eV/Angstrom^2"
    ifc_ref_energy.unit="eV"

    ref_masses[:] = atoms.get_masses()
    ref_zion[:] = atoms.get_atomic_numbers()
    ref_cell[:] = atoms.get_cell()
    ref_xred[:] = atoms.get_scaled_positions()
    ref_xcart[:] = atoms.get_positions()
    ifc_ref_energy[:]=ref_energy

    ifc_Rlist[:] = np.array(Rlist)
    ifc_vallist[:] = np.real(ifc_vals)
    ifc_vallist_imag[:] = np.imag(ifc_vals)
    root.close()


def read_ifc_from_netcdf(fname):
    """
    read netcdf file to atoms, Rlist, and ifc values
    """
    root = Dataset(fname, 'r')
    ref_masses = root.variables['ref_masses'][:]
    ref_xred = root.variables['ref_xred'][:]
    ref_cell = root.variables['ref_cell'][:]
    ref_zion = root.variables['ref_zion'][:]
    atoms = Atoms(numbers=ref_zion,
                  cell=ref_cell,
                  scaled_positions=ref_xred,
                  masses=ref_masses)
    Rlist = root.variables['ifc_Rlist'][:]
    ifc_vallist = root.variables['ifc_vallist'][:] + root.variables['ifc_vallist_imag'][:]*1.0j
    return atoms, Rlist, ifc_vallist
