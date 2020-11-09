import logging
logging.basicConfig(filename=f'collect.log', filemode='w',
                                          level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

import numpy as np
from ase.atoms import Atoms

from thyme.routines.folders import parse_merged_folders_trjs
from thyme.parsers.vasp import pack_folder_trj, get_childfolders
from thyme.routines.dist_plots.energy import multiple_plots as multiple_plots_e
from thyme.filters.distance import e_filter
from thyme.trajectories import Trajectories

def main():

    folders = get_childfolders("./")
    trjs = parse_merged_folders_trjs(folders, pack_folder_trj, e_filter, "all_data.pickle")
    trjs.save("alldata_padded_mat.npz")
    trj = trjs.to_padded_trajectory()
    trjs = Trajectories.from_padded_trajectory(trj)
    multiple_plots_e(trjs, prefix='alldata')

if __name__ == '__main__':
    main()
