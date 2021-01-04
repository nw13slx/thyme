from os.path import isfile

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
logging.basicConfig(filename=f'collect_merge.log', filemode='w',
                    level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())


def main():

    if not isfile("all_data.pickle"):
        folders = get_childfolders("./")
        trjs = parse_merged_folders_trjs(folders, pack_folder_trj,
                                  e_filter, npz_filename="all_data.pickle", merge_level=1)
    else:
        trjs = Trajectories.from_file('all_data.pickle')

    trjs = trjs.remerge()
    for name, trj in trjs.alldata.items():
        frames = sort_e(trj)
        trj.filter_frames(frames)
        mine = trj.energies[0]
        keep_id = np.where(trj.energies < (mine+10))[0]
        trj.filter_frames(keep_id)
    multiple_plots_e(trjs, prefix='alldata')
    write_trjs("all.xyz", trjs)


if __name__ == '__main__':
    main()
