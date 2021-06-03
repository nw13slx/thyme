import logging
import numpy as np

from ase.atoms import Atoms
from ase.io.extxyz import write_xyz


def write_single_xyz(trj, prefix):
    """"""
    for i in range(trj.nframes):
        atoms = Atoms(trj.species, trj.positions[i], cell=trj.cells[i], pbc=True)
        write_xyz(f"{prefix}.xyz", atoms, append=True)


def write_xyzs(trjs, prefix=""):
    """"""
    for trj in trjs.alltrjs.values():
        write_single_xyz(trj, f"{prefix}{trj.name}")
