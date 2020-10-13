import numpy as np
from copy import deepcopy

class Trajectory():

    per_frame_keys = ["positions", "forces", "energies",
                      "cells"]
    metadata_keys = ["dipole_correction", "species",
                "nelm", "nframes", "cutoff",
                "natom", "kpoints"]

    def __init__(self):
        """
        dummy init. do nothing
        """

        allkeys = Trajectory.per_frame_keys + Trajectory.metadata_keys
        for k in allkeys:
            setattr(self, k, None)

        self.nframes = 0
        self.per_frame_attrs = deepcopy(Trajectory.per_frame_keys)
        self.metadata_attrs = deepcopy(Trajectory.metadata_keys)

        self.python_list = False
        self.empty = True

    @staticmethod
    def from_dict(dictionary):
        trj = Trajectory()
        for k in dictionary:
            if k in default_key:
                setattr(trj, k, np.copy(dictionary[k]))
            else:
                raise NotImplementedError("add numpy arrays")
        return trj

    def add_containers(self, natom: int=0,
                       species=None,
                       attributes: list =None):
        """
        initialize all attributes with empty list (python_list = True)
        or numpy array (python_list = False)
        """

        if self.python_list:

            if attributes is not None:
                for k in attributes:
                    self.per_frame_attrs.append(k)

            for k in self.per_frame_attrs:
                setattr(self, k, [])

        else:
            raise NotImplementedError("add numpy arrays")

        self.natom = natom
        self.species = species
        self.empty = False

    def add_frame_from_dict(self, dictionary:dict, nframes:int,
                            i:int=-1, attributes:list=None, idorder=None):
        """
        add one(i) or all frames from dictionary to trajectory
        """

        if i < 0:
            self.add_frames_from_dict(dictionary=dictionary, nframes=nframes,
                            attributes=attributes, idorder=iorder)
            return

        natom = len(idorder)
        species=dictionary['symbols'][i]

        if idorder is not None:
            species=species[idorder],

        if self.empty:
            self.add_containers(natom=natom,
                                species=species,
                                attributes=attributes)

        if i > 0:
            for k in self.per_frame_attrs:
                if k in dictionary:
                    dim = len(dictionary[k].shape)
                    if dim == 1:
                        getattr(self, k).append(dictionary[k][i])
                    elif dim >= 2:
                        if dictionary[k].shape[1] > natom:
                            if idorder is not None:
                                getattr(self, k).append(dictionary[k][i][idorder])
                            else:
                                getattr(self, k).append(dictionary[k][i])
                        else:
                            getattr(self, k).append(dictionary[k][i])
                else:
                    raise RuntimeError(f"{k} is needed")

    def add_frames_from_dict(self, dictionary:dict, nframes:int,
                             attributes:list=None, idorder=None):
        """
        add one(i) or all frames from dictionary to trajectory
        """

        natom = dictionary['symbols'].shape[1]
        species=dictionary['symbols'][0]

        if idorder is not None:
            species=species[idorder],

        if self.empty:
            self.python_list = False
            self.add_containers(natom=natom,
                                species=species,
                                attributes=attributes)
        elif self.python_list:
            self.convert_to_np()

        raise NotImplementedError("add numpy arrays")


    def convert_to_np(self):
        """
        assume all elements are the same in the list
        """

        if not self.python_list:
            return

        for k in self.per_frame_attrs:
            np_mat = np.array(getattr(self, k))
            setattr(self, k, np_mat)

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

    per_frame_keys = ["natoms", "symbols"]
    per_frame_keys.append(Trajectory.per_frame_keys)

    def __init__(self):
        Trajectory.__init__(self)

    def padd(self, max_atoms):

        datom = max_atom-self.natoms

        if datom > 0:
            for k in ['positions', 'forces', 'symbols', 'pred']:

                new_shape = np.copy(data[k].shape)
                dim = new_shape[1]//natoms
                new_shape[1] = dim*max_atom-new_shape[1]
                pad = np.zeros(new_shape)
                data[k] = np.hstack([data[k], pad])
