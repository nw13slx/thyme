import logging
import numpy as np

from glob import glob
from os.path import getmtime, isfile

from thyme.trajectory import Trajectory
from thyme.trajectories import Trajectories
from thyme.routines.folders import find_folders_matching
from thyme.parsers.extxyz import extxyz_to_padded_trj
from thyme.parsers.extxyz import pack_folder as pack_extxyz_folder
from thyme.parsers.extxyz import pack_folder_trj as pack_extxyz_folder_trj

from ase.atoms import Atoms


def get_childfolders(path, include_xyz=True):

    return find_folders_matching(['*_*.xyz'], path)

def pack_folder_trj(folder, data_filter, include_xyz=True):

    trjs = Trajectories()
    for filename in glob(f"{folder}/*_*.xyz"):
        if isfile(filename):
            trj = extxyz_to_padded_trj(filename)
            if trj.nframes > 0:
                trjs.add_trj(trj, filename)
    return trjs
