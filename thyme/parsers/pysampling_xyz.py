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
            labeling(trj)
            if trj.nframes > 0:
                trjs.add_trj(trj, filename)
    return trjs

def labeling(trj):

    species = trj.symbols[-1]
    atoms = Atoms(species, trj.positions[-1], cell=trj.cells[-1], pbc=True)

    Hid = np.array([i for i, s in enumerate(species) if s == 'H'])
    Cid = np.array([i for i, s in enumerate(species) if s == 'C'])
    Oid = np.array([i for i, s in enumerate(species) if s == 'O'])
    Cuid = np.array([i for i, s in enumerate(species) if s == 'Cu'])

    dist_mat = atoms.get_all_distances(mic=True)
    dist_CH = np.max((dist_mat[Hid].T)[Cid])

    dist_OCu = np.max(np.min((dist_mat[Oid].T)[Cuid], axis=0))
    dist_HCu = np.min((dist_mat[Hid].T)[Cuid])

    label = -1
    if dist_CH > 2.5 and dist_OCu > 3 and dist_HCu < 1.8:
        label = 1
    elif dist_CH <= 1.2 and dist_OCu < 2.2 and dist_HCu > 3:
        label = 0

    trj.labels = np.zeros(trj.nframes) + label
    trj.per_frame_attrs += ['labels']
    logging.info(f"label as {label}")
