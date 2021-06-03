import logging
import numpy as np

from collections import Counter
from glob import glob
from os.path import getmtime, isfile

from thyme.trajectory import Trajectory
from thyme.trajectories import Trajectories
from thyme.routines.folders import find_folders_matching


def get_childfolders(path, include_xyz=True):

    return find_folders_matching(["assm_pth*_*.npz"], path)


def pack_folder_trj(folder, data_filter, include_xyz=True):

    trjs = Trajectories()
    for filename in glob(f"{folder}/assm_pth*_*.npz"):
        if isfile(filename):
            trj = parse_npz(filename, data_filter)
            if trj.nframes > 0:
                trjs.add_trj(trj, filename)
    return trjs


def parse_npz(filename, data_filter):

    trj = Trajectory()
    logging.info(f"parsing {filename} as npz")
    dictionary = dict(np.load(filename))
    for k in ["alle", "x", "intc", "n"]:
        if k not in dictionary:
            return Trajectory()

    nframes = len(dictionary["x"])
    trj.position = dictionary["x"].reshape([nframes, -1, 3])
    trj.energies = dictionary["alle"][:, 1]
    trj.pe = dictionary["intc"][:, -1]
    trj.intc = dictionary["intc"][:, :-1]
    trj.per_frame_attrs = ["positions", "energies", "pe", "intc"]
    trj.sanity_check()

    if data_filter is not None:
        ids = data_filter(trj)
        trj.include_frames(ids)

    logging.info(f"trj {trj}")
    return trj
