import logging
import numpy as np
import pickle

from glob import glob
from os.path import getmtime

from flare.struc import Structure

from thyme.parsers.monty import read_pattern, read_table_pattern
from thyme.trajectory import PaddedTrajectory, Trajectory
from thyme.routines.folders import find_folders, find_folders_matching


def write(filename, trj):
    structures = []
    if isinstance(trj, Trajectory) and not isinstance(trj, PaddedTrajectory):
        for i in range(trj.nframes):
            natom = trj.natoms[i]
            structure = Structure(
                cell=trj.cells[i].reshape([3, 3]),
                species=trj.species[:natom],
                positions=trj.positions[i][:natom].reshape([-1, 3]),
                forces=trj.forces[i][:natom].reshape([-1, 3]),
                energy=trj.energies[i],
            )
            structures += [structure]
    elif isinstance(trj, PaddedTrajectory):
        for i in range(trj.nframes):
            structure = Structure(
                cell=trj.cells[i].reshape([3, 3]),
                species=trj.symbols[i],
                positions=trj.positions[i].reshape([-1, 3]),
                forces=trj.forces[i].reshape([-1, 3]),
                energy=trj.energies[i],
            )
            structures += [structure]

    with open(filename, "wb") as fout:
        pickle.dump(structures, fout)

    return structures
