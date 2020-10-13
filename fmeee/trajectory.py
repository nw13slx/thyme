import numpy as np

class Trajectory():

    default_keys = ["positions", "forces", "energies",
                    "cells", "nelm", "nframes", "cutoff",
                    "dipole_correction", "species",
                    "natom", "kpoints", "species"]

    def __init__(self):

        positions = None
        forces = None
        energies = None
        cells = None
        nelm = None
        cutoff = None
        dipole_correction = None
        species = None
        kpoints = None
        nframes = 0
        natoms = None

    def from_dict(self, dictionary):

        for k in dictionary:
            if k in default_key:
                setattr(self, k, np.copy(dictionary[k]))
            else:
                pass


    def reshape(self):

        if len(self.positions.shape) == 2:
            self.positions = self.positions.reshape([self.nframes, self.natoms, 3])
        if len(self.forces.shape) == 2:
            self.forces = self.forces.reshape([self.nframes, self.natoms, 3])
        if len(self.cells.shape) == 2:
            self.cells = self.cells.reshape([self.nframes, 3, 3])

    def flatten(self):

        if len(self.positions.shape) == 3:
            self.positions = self.positions.reshape([self.nframes, -1])
        if len(self.forces.shape) == 3:
            self.forces = self.forces.reshape([self.nframes, -1])
        if len(self.cells.shape) == 3:
            self.cells = self.cells.reshape([self.nframes, 9])


class PaddedTrajectory(Trajectory):

    default_keys = ["natoms", "symbols"]
    default_keys.append(Trajectory.default_keys)

    def __init__(self):
        Trajectory.__init__(self)

    def padd(self, max_atoms):

        datom = max_atom-self.natoms

        print(list(data.keys()))
        if datom > 0:
            for k in ['positions', 'forces', 'symbols', 'pred']:

                new_shape = np.copy(data[k].shape)
                dim = new_shape[1]//natoms
                new_shape[1] = dim*max_atom-new_shape[1]
                pad = np.zeros(new_shape)
                data[k] = np.hstack([data[k], pad])

class Trajectories():

    def __init__(self):
        pass
