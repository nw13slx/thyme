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
    trjs = parse_folders_trjs(folders, pack_folder_trj, e_filter, "all_data_raw.pickle")
    # trjs.save("raw.pickle")

    trjs = trjs.remerge()
    trjs.save("all_data_merge.pickle")

    mineT = Trajectories()
    for name, trj in trjs.alldata.items():
        # sort by energy
        frames = sort_e(trj)
        trj.include_frames(frames)
        mine = trj.energies[0]
        keep_id = np.where(trj.energies < (mine + 20))[0]
        mineT.add_trj(trj.extract_frames([-1]))
    # multiple_plots_e(trjs, prefix='alldata')
    # frames = sort_e(mineT)
    # mineT.include_frames(frames)
    mineT.save("20eV.pickle")
    # write("mine.xyz", mineT)
    write_trjs("all.xyz", trjs)


if __name__ == "__main__":
    main()
