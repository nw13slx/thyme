import logging
import numpy as np

from glob import glob
from os.path import getmtime

from flare.struc import Structure

from fmeee.parsers.monty import read_pattern, read_table_pattern
from fmeee.trajectory import PaddedTrajectory
from fmeee.routines.folders import find_folders, find_folders_matching

def write(trj):
    structures = []
    if isinstance(trj, Trajectory) and not isinstance(trj, PaddedTrajectory):
        for i in range(trj.nframes):
            natom = trj.natoms[i]
            structure = Structure(cell=trj.cells[i].reshape([3, 3]),
                                  species=trj.species[:natom],
                                  positions=trj.positions[i][:natom].reshape([-1, 3]),
                                  forces=trj.forces[i][:natom].reshape([-1, 3]),
                                  energiy=trj.energy[i])
            structures += [structure]
    elif isinstance(trj, PaddedTrajectory):
        for i in range(trj.nframes):
            structure = Structure(cell=trj.cells[i].reshape([3, 3]),
                                  species=trj.symbols[i],
                                  positions=trj.positions[i].reshape([-1, 3]),
                                  forces=trj.forces[i].reshape([-1, 3]),
                                  energiy=trj.energy[i])
            structures += [structure]
    return structures
