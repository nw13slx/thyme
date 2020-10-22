import logging
import numpy as np

from glob import glob
from os.path import getmtime

from fmeee.parsers.monty import read_pattern, read_table_pattern
from fmeee.parsers.extxyz import extxyz_to_padded_trj
from fmeee.parsers.extxyz import pack_folder as pack_extxyz_folder
from fmeee.trajectory import PaddedTrajectory
from fmeee.routines.folders import find_folders_matching

def get_childfolders(path, include_xyz=True):

    if include_xyz:
        return find_folders_matching(['xyz_strucs/*.xyz', 'xyz_strucs/*.extxyz'], path)
    else:
        return find_folders_matching(['xyz_strucs/*.extxyz'], path)

def pack_folder(folder, data_filter, include_xyz=True):

    return pack_extxyz_folder(f"{folder}/xyz_strucs", data_filter, include_xyz)
