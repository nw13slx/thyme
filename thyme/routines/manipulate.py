from thyme.trajectory import Trajectory
import numpy as np


def replicate(trj, expand: tuple = (1, 1, 1)):

    new_trj = Trajectory()
    for idf in range(len(trj)):
        _trj = trj.skim([idf])
        cell = _trj.cells[0]
        new_positions = []
        new_cells = []
        new_forces = []
        symbols = []
        for i in range(expand[0]):
            for j in range(expand[1]):
                for k in range(expand[2]):
                    new_positions.append(
                        _trj.positions[0]
                        + cell[0] * i
                        + cell[1] * j
                        + cell[2] * k
                    )
                    new_forces.append(np.copy(_trj.forces[0]))
                    symbols.append(np.copy(_trj.species))
        new_positions = np.vstack(new_positions)
        new_forces = np.vstack(new_forces)
        symbols = np.hstack(symbols)
        cell = np.copy(_trj.cells[0])
        for idir in range(3):
            for jdir in range(3):
                cell[idir, jdir] = cell[idir, jdir] * expand[idir]
        _trj.positions = new_positions.reshape([1, -1, 3])
        _trj.forces = new_forces.reshape([1, -1, 3])
        _trj.cells = cell.reshape([1, 3, 3])
        _trj.natom = expand[0] * expand[1] * expand[2] * _trj.natom
        new_trj.add_trj(_trj)
    return new_trj
