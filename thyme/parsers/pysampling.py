import logging
import numpy as np

from collections import Counter
from glob import glob
from os.path import getmtime, isfile

from thyme.trajectory import Trajectory
from thyme.routines.folders import find_folders_matching


def get_childfolders(path, include_xyz=True):

    return find_folders_matching(['*_*.npz'], path)

def pack_folder_trj(folder, data_filter, include_xyz=True):

    trj = Trajectory()
    for filename in glob(f"{folder}/*_*.npz"):
        if isfile(filename):
            _trj = parse_npz(filename, data_filter)
            if _trj.nframes > 0:
                trj.add_trj(_trj)
    return trj


def parse_npz(filename, data_filter):

    logging.info(f"parsing {filename} as npz")
    dictionary = dict(np.load(filename))
    for k in ['alle', 'x', 'intc', 'n', 'sp']:
        if k not in dictionary:
            return Trajectory()

    nframes = len(dictionary['x'])
    dictionary['positions'] = dictionary['x'].reshape([nframes, -1, 3])
    dictionary['energies'] = dictionary['alle'][:, 1]
    dictionary['H_energy'] = dictionary['intc'][:, -1]
    dictionary['intc'] = dictionary['intc'][:, :-1]

    del dictionary['alle']
    del dictionary['x']
    del dictionary['v']

    sp = int(dictionary['sp'])
    n1 = int(dictionary['n'][0])
    n2 = int(dictionary['n'][1])

    dictionary['labels'] = np.zeros(nframes)
    dictionary['labels'][:sp] += n1
    dictionary['labels'][sp:] += n2
    c = Counter(dictionary['labels'])

    logging.info(f"label the track {sp}/{nframes} {n1} vs {n2} {c}")

    dictionary['filename'] = filename

    trj = Trajectory.from_dict(dictionary)
    if data_filter is not None:
        ids = data_filter(trj)
        trj.filter_frames(ids)
    return trj

