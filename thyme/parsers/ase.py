import logging
import numpy as np

from math import inf

from ase.atoms import Atoms
from thyme.trajectory import Trajectory, PaddedTrajectory


def configuration(trj, i):
    if isinstance(trj, PaddedTrajectory):
        natom = trj.natoms[i]
        structure = Atoms(
            cell=trj.cell[i].reshape([3, 3]),
            symbols=trj.symbols[i][:natom],
            positions=trj.position[i][:natom].reshape([natom, 3]),
            pbc=True,
        )
    else:
        structure = Atoms(
            cell=trj.cells[i].reshape([3, 3]),
            symbols=trj.symbols[i],
            positions=trj.positions[i].reshape([-1, 3]),
            pbc=True,
        )
    return structure
