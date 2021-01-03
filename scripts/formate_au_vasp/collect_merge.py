from thyme.filters.distance import e_filter
from thyme.filters.energy import sort_e
from thyme.routines.dist_plots.energy import multiple_plots as multiple_plots_e
from thyme.parsers.extxyz import write_trjs
from thyme.parsers.vasp import pack_folder_trj, get_childfolders
from thyme.routines.folders import parse_merged_folders_trjs
from thyme.trajectories import Trajectories
from ase.atoms import Atoms
import numpy as np
import logging
logging.basicConfig(filename=f'collect.log', filemode='w',
                    level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())


def main():

    # folders = get_childfolders("./")
    # # folders = [f"{i+1}" for i in range(59)]
    # trjs = parse_merged_folders_trjs(folders, pack_folder_trj,
    #                           e_filter, "all_data.pickle")
    trjs = Trajectories.from_file('all_data.pickle')
    # merge the one with the same composition
    trjs = Trajectories.from_padded_trajectory(trjs.to_padded_trajectory())
    for trj in trjs.alldata.values():
        accept_id = sort_e(trj)
        trj.filter_frames(accept_id)
        write_trjs("all.xyz", trjs)

    multiple_plots_e(trjs, prefix='alldata')


if __name__ == '__main__':
    main()
