from netCDF4 import Dataset


def read_wannc(fname):
    ds=Dataset(fname)
    print(ds.variables)


read_wannc('./wann.hdf')
