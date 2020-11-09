from thyme.trajectories import Trajectories
from thyme.trajectory import PaddedTrajectory
from thyme.parsers.extxyz import extxyz_to_padded_trj
from thyme.routines.folders import parse_folders
from ase.atoms import Atoms
from glob import glob
import numpy as np
import logging
logging.basicConfig(filename=f'collect.log', filemode='w',
                    level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())


def main():

    join_trj = PaddedTrajectory()
    for filename in glob("*.xyz"):
        join_trj.add_trj(extxyz_to_padded_trj(filename))
    join_trj.save(f"{join_trj.nframes}frames.npz")
#     folders = get_childfolders("./")
#     parse_folders(folders, pack_folder, e_filter, "all_data")


if __name__ == '__main__':
    main()
