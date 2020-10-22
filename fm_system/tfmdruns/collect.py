import logging
logging.basicConfig(filename=f'collect.log', filemode='w',
                                          level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

import numpy as np
from ase.atoms import Atoms

from fmeee.routines.folders import parse_folders_trjs
from fmeee.parsers.tfmd import pack_folder_trj, get_childfolders
from fmeee.routines.dist_plots.energy import multiple_plots as multiple_plots_e

def main():

    folders = get_childfolders("./", include_xyz = True)
    trjs = parse_folders_trjs(folders, pack_folder_trj, e_filter, "all_data.pickle")
    trjs.save("alldata_padded_mat.npz")
    multiple_plots_e(trjs, prefix='alldata')

def e_filter(trj):
    print("enter efilter")
    set1 = set(np.where(trj.energies<0)[0])
    set2 = set(np.where(trj.energies>-1000)[0])
    return list(set1.intersection(set2))

if __name__ == '__main__':
    main()
