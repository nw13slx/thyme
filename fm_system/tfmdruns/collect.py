import logging
logging.basicConfig(filename=f'collect.log', filemode='w',
                                          level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

import numpy as np
from ase.atoms import Atoms

from fmeee.routines.folders import parse_folders_trjs
from fmeee.parsers.tfmd import pack_folder_trj, get_childfolders

def main():

    folders = get_childfolders("./", include_xyz = True)
    trjs = parse_folders_trjs(folders, pack_folder_trj, e_filter, "all_data.pickle")
    trjs.save("alldata_padded_mat.npz")

def e_filter(xyz, f, e, c, species):
    return True

if __name__ == '__main__':
    main()
