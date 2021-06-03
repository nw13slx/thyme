from thyme.trajectories import Trajectories
from thyme.trajectory import Trajectory
from thyme.filters.distance import e_filter
from thyme.filters.energy import sort_e
from thyme.parsers.extxyz import write_trjs
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
    trjs = parse_folders_trjs(folders, pack_folder_trj, e_filter, "all_data_raw.npz")

    trjs.merge()
    trjs.save("all_data_merge.npz")

    mineT = Trajectories()
    for name, trj in trjs.alltrjs.items():
        # sort by energy
        frames = sort_e(trj)
        trj.include_frames(frames)
        mine = trj.total_energy[0]
        keep_id = np.where(trj.total_energy < (mine + 20))[0]
        mineT.add_trj(trj.extract_frames([-1]))
    # multiple_plots_e(trjs, prefix='alldata')
    # frames = sort_e(mineT)
    # mineT.include_frames(frames)
    mineT.save("20eV.npz")
    write_trjs("all.xyz", trjs, joint=True)


if __name__ == "__main__":
    main()
