import logging
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
        self.python_list = False
        self.empty = True
        self.name = ""

        self.per_frame_attrs = []
        self.metadata_attrs = ['nframes', 'name', 'python_list', 'empty']

    def __repr__(self):
        s = f"{self.name}: {self.nframes} frames with {self.natom} atoms\n"
        for k in self.per_frame_attrs:
            s+= f"{k} "
        s += "\n"
        for k in self.metadata_attrs:
            s+= f"{k} "
        return s

    def sanity_check(self):

        if len(self.per_frame_attrs) != 0:

            self.empty = False
            frames = []
            for k in self.per_frame_attrs:
                frames += [len(getattr(self, k))]

            if len(set(frames)) > 1:
                raise RuntimeError("Data inconsistent")

            self.nframes = frames[0]
            self.per_frame_attrs = list(set(self.per_frame_attrs))

        self.metadata_attrs = list(set(self.metadata_attrs))


    @staticmethod
    def from_dict(dictionary):
        trj = Trajectory()
        for k in dictionary:
            if k in Trajectory.metadata_keys:
                setattr(trj, k, np.copy(dictionary[k]))
                trj.per_frame_attrs += [k]
            elif k in Trajectory.metadata_keys:
                setattr(trj, k, np.copy(dictionary[k]))
                trj.metadata_attrs += [k]
            else:
                raise NotImplementedError("add numpy arrays")
        trj.sanity_check()
        return trj

    def save(self, name: str, format: str = None):

        supported_formats = ['pickle', 'npz'] # npz

        for detect in supported_formats:
            if detect in name.lower():
                format = detect
                break

        if format is None:
            format = supported_formats[0]
        format = format.lower()
        if f'{format}' != name[-len(format):]:
            name += f'.{format}'

        if format == 'pickle':
            with open(name, 'wb') as f:
                pickle.dump(self, f)
        elif format == 'npz':
            if '.npz' != name[-4:]:
                name += '.npz'
            for k in self.__dict__:
                s = f"write {k}"
                try:
                    s += "{self.__dict__[k].shape}"
                except:
                    s += f"{self.__dict__[k]}"
                logging.debug(s)
            np.savez(name, **self.__dict__)
            logging.info(f"save as {name}")
        else:
            raise NotImplementedError(f"Output format not supported:"
                                      f" try from {supported_formats}")

    def add_containers(self, natom: int=0,
                       species=None,
                       attributes: list =None):
        """
        initialize all attributes with empty list (python_list = True)
        or numpy array (python_list = False)

        attributes: only per_frame_attrs needs to be listed
        """

        if self.python_list:

            if attributes is not None:
                for k in attributes:
                    if k not in self.per_frame_attrs:
                        self.per_frame_attrs.append(k)

            for k in self.per_frame_attrs:
                setattr(self, k, [])

        else:
            raise NotImplementedError("add numpy arrays")

        self.natom = natom
        self.species = species
        self.empty = False
        for k in ['natom', 'species']:
            if k not in self.metadata_attrs:
                self.metadata_attrs.append(k)

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
        ori_natom = len(species)

        if idorder is not None:
            species=[species[i] for i in idorder]

        if self.empty:
            self.add_containers(natom=natom,
                                species=species,
                                attributes=attributes)

        for k in self.per_frame_attrs:
            if k in dictionary:
                dim = len(dictionary[k].shape)
                if dim == 1:
                    getattr(self, k).append(dictionary[k][i])
                elif dim >= 2:
                    if dictionary[k].shape[1] == ori_natom:
                        if idorder is not None:
                            getattr(self, k).append(dictionary[k][i][idorder])
                        else:
                            getattr(self, k).append(dictionary[k][i])
                    elif dictionary[k].shape[1] > ori_natom:
                        raise RuntimeError(f"{k} {dictionary[k].shape} {ori_natom} "
                                           "cannot be handled")
                    else:
                        getattr(self, k).append(dictionary[k][i])
            else:
                raise RuntimeError(f"{k} is needed")
        self.nframes += 1

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
        self.convert_to_np()

        raise NotImplementedError("add numpy arrays")

    def add_trj(self, trj):
        """
        add all frames from another trajectory instance
        """

        if self.empty:
            self.copy(trj)
            self.convert_to_np()
        else:

            self.convert_to_np()

            assert self.natom == trj.natom

            for k in self.per_frame_attrs:
                item = getattr(trj, k)
                ori_item = getattr(self, k)
                if len(item.shape) == 1:
                    setattr(self, k, np.hstack((ori_item, item)))
                else:
                    setattr(self, k, np.vstack((ori_item, item)))
                ori_item = getattr(self, k)
            self.nframes += trj.nframes

    def convert_to_np(self):
        """
        assume all elements are the same in the list
        """

        if not self.python_list:
            return

        for k in self.per_frame_attrs:
            np_mat = np.array(getattr(self, k))
            if np_mat.shape[0] != self.nframes:
                raise RuntimeError(f"inconsistent content {np_mat.shape} {k}"
                                   f" and counter {self.nframes}")
            logging.debug(f"convert content {k} to numpy array"
                          f" with shape {np_mat.shape} ")
            setattr(self, k, np_mat)
        self.python_list = False

    def convert_to_list(self):
        """
        assume all elements are the same in the list
        """

        if self.python_list:
            return

        for k in self.per_frame_attrs:
            list_mat = [i for i in getattr(self, k)]
            setattr(self, k, list_mat)
        self.python_list = True

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

    def skim(self, frame_list):

        self.convert_to_np()

        trj = Trajectory()
        trj.empty = False
        trj.nframes = len(frame_list)

        for k in self.per_frame_attrs:
            setattr(trj, k, getattr(self, k)[frame_list])
        for k in self.metadata_attrs:
            setattr(trj, k, getattr(self, k))
        trj.python_list = False
        trj.name = f"{self.name}_{trj.nframes}"
        trj.sanity_check()

        return trj

    def copy(self, otrj):

        for k in otrj.per_frame_attrs:
            setattr(self, k, deepcopy(getattr(otrj, k)))
            self.per_frame_attrs += [k]
        for k in otrj.metadata_attrs:
            setattr(self, k, deepcopy(getattr(otrj, k)))
            self.metadata_attrs += [k]
        self.empty = otrj.empty
        self.sanity_check()


class PaddedTrajectory(Trajectory):

    per_frame_keys = ["natoms", "symbols"]
    per_frame_keys.append(Trajectory.per_frame_keys)

    def __init__(self):
        Trajectory.__init__(self)

    def sanity_check(self):
        Trajectory.sanity_check(self)
        if 'species' in self.metadata_attrs:
            del self.species
            self.metadata_attrs.remove('species')

    @staticmethod
    def from_trajectory(otrj, max_atom):

        otrj.convert_to_np()
        otrj.sanity_check()

        datom = max_atom-otrj.natom
        trj = PaddedTrajectory()
        if datom == 0:
            trj.copy(otrj)
        elif datom < 0:
            raise RuntimeError("wrong max atom is set")

        else:

            trj.per_frame_attrs = []
            for k in otrj.per_frame_attrs:

                trj.per_frame_attrs += [k]
                item = getattr(otrj, k)
                dim = len(item.shape)

                if dim == 1:
                    setattr(trj, k, np.copy(item))
                elif dim >=2:

                    if item.shape[1] == otrj.natom:
                        new_shape = np.copy(item.shape)
                        new_shape[1] = max_atom-new_shape[1]
                        pad = np.zeros(new_shape)
                        new_item = np.hstack([item, pad])
                        setattr(trj, k, new_item)
                    else:
                        setattr(trj, k, np.copy(item))
                else:
                    raise RuntimeError(f"{k} is needed")

            trj.metadata_attrs = []
            for k in otrj.metadata_attrs:
                if k != 'species':
                    setattr(trj, k, getattr(otrj, k))
                    trj.metadata_attrs += [k]

            species = np.hstack([otrj.species, datom*['NA']])
            trj.symbols = np.vstack([species]*trj.nframes)
            trj.natoms = np.ones(trj.nframes)*otrj.natom
            trj.per_frame_attrs += ['symbols']
            trj.per_frame_attrs += ['natoms']
            trj.name = f"{otrj.name}_padded"
            trj.natom = max_atom

        trj.sanity_check()
        logging.debug(f"! return {trj}")

        return trj

