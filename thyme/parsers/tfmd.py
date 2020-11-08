import logging
import numpy as np

from glob import glob
from os.path import getmtime

from thyme.parsers.monty import read_pattern, read_table_pattern
from thyme.parsers.extxyz import extxyz_to_padded_trj
from thyme.parsers.extxyz import pack_folder as pack_extxyz_folder
from thyme.parsers.extxyz import pack_folder_trj as pack_extxyz_folder_trj
from thyme.trajectory import PaddedTrajectory
from thyme.routines.folders import find_folders_matching

def get_childfolders(path, include_xyz=True):

    if include_xyz:
        return find_folders_matching(['xyz_strucs/*.xyz', 'xyz_strucs/*.extxyz'], path)
    else:
        return find_folders_matching(['xyz_strucs/*.extxyz'], path)

def pack_folder(folder, data_filter, include_xyz=True):

    return pack_extxyz_folder(f"{folder}/xyz_strucs", data_filter, include_xyz)

def pack_folder_trj(folder, data_filter, include_xyz=True):

    return pack_extxyz_folder_trj(f"{folder}/xyz_strucs", data_filter, include_xyz)
