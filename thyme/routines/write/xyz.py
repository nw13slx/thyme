import logging
import numpy as np

from ase.atoms import Atoms
from ase.io.extxyz import write_xyz

from thyme.trajectory import PaddedTrajectory

def write_single_xyz(trj, prefix):
    """
    """
    if isinstance(trj, PaddedTrajectory):
        for i in range(trj.nframes):
            atoms = Atoms(trj.symbols[i], trj.positions[i], cell=trj.cells[i], pbc=True)
            write_xyz(f"{prefix}.xyz", atoms, append=True)
    else:
        for i in range(trj.nframes):
            atoms = Atoms(trj.species, trj.positions[i], cell=trj.cells[i], pbc=True)
            write_xyz(f"{prefix}.xyz", atoms, append=True)

def write_xyzs(trjs, prefix=""):
    """
    """
    for trj in trjs.alldata.values():
        write_single_xyz(trj, f"{prefix}{trj.name}")
