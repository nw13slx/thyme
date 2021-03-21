from thyme.trajectories import Trajectories
from thyme.trajectory import Trajectory
from thyme.filters.distance import e_filter
from thyme.filters.energy import sort_e
from thyme.parsers.extxyz import write_trjs, write
from thyme.routines.dist_plots.energy import multiple_plots as multiple_plots_e
from thyme.parsers.vasp import pack_folder_trj, get_childfolders
from thyme.routines.folders import parse_folders_trjs
from ase.atoms import Atoms
import numpy as np
import logging

logging.basicConfig(
    filename=f"collect.log", filemode="w", level=logging.DEBUG, format="%(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())


def main():

    folders = get_childfolders("./")
    trjs = parse_folders_trjs(folders, pack_folder_trj, e_filter, "all_data.pickle")

    mineT = Trajectory()
    for name, trj in trjs.alldata.items():
        frames = sort_e(trj)
        trj.filter_frames(frames)
        mine = trj.energies[0]
        keep_id = np.where(trj.energies < (mine + 10))[0]
        trj.filter_frames(keep_id)
        mineT.add_trj(trj.skim([-1]))
    # multiple_plots_e(trjs, prefix='alldata')
    frames = sort_e(mineT)
    mineT.filter_frames(frames)
    write("mine.xyz", mineT)
    write_trjs("all.xyz", trjs)


if __name__ == "__main__":
    main()
