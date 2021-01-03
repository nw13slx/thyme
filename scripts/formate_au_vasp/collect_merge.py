from thyme.filters.distance import e_filter
from thyme.routines.dist_plots.energy import multiple_plots as multiple_plots_e
from thyme.parsers.extxyz import write_trjs
from thyme.parsers.vasp import pack_folder_trj, get_childfolders
from thyme.routines.folders import parse_merged_folders_trjs
from thyme.trajectories import Trajectories
from ase.atoms import Atoms
import numpy as np
import logging
logging.basicConfig(filename=f'collect_merge.log', filemode='w',
                    level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())


def main():

    # folders = get_childfolders("./")
    # folders = [f"{i+1}" for i in range(59)]
    # trjs = parse_merged_folders_trjs(folders, pack_folder_trj,
    #                           e_filter, npz_filename="all_data.pickle", merge_level=1)
    trjs = Trajectories.from_file('all_data.pickle')
    trjs = trjs.remerge()
    multiple_plots_e(trjs, prefix='alldata')
    write_trjs("all.xyz", trjs)


if __name__ == '__main__':
    main()
