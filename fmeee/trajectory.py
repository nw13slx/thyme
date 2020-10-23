import logging
import numpy as np
from copy import deepcopy

from ase.io.extxyz import write_xyz as write_extxyz
from ase.io.vasp import write_vasp
from ase.atoms import Atoms

from fmeee.utils.cell import convert_cell_format
from fmeee.utils.save import sort_format

class Trajectory():

    per_frame_keys = ["positions", "forces", "energies",
                      "cells"]
    metadata_keys = ["dipole_correction", "species",
                     "nelm", "nframes", "cutoff",
                     "natom", "kpoints", "empty",
                     "python_list", "name"]

    def __init__(self):
        """
        dummy init. do nothing
        """

        allkeys = type(self).per_frame_keys + type(self).metadata_keys
        for k in allkeys:
            setattr(self, k, None)

        self.nframes = 0
        self.python_list = False
        self.empty = True
        self.name = ""

        self.per_frame_attrs = []
        self.metadata_attrs = ['nframes', 'name', 'python_list', 'empty']

    def __repr__(self):
        s = f"{self.name}: {self.nframes} frames with {self.natom} atoms"
        return s

    def __str__(self):
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

            for k in self.per_frame_attrs:
                item = getattr(self, k)
                try:
                    if item.shape[1] % self.natom == 0:
                        item = item.reshape([self.nframes, ori_natom, -1])
                        setattr(self, k, item)
                except:
                    pass

        self.metadata_attrs = list(set(self.metadata_attrs))

    def filter_frames(self, accept_id=None):

        if accept_id is None:
            return

        for k in self.per_frame_attrs:
            new_mat = getattr(self, k)[accept_id]
            setattr(self, k, new_mat)
        self.nframes = len(accept_id)


    @staticmethod
    def from_file(filename):
        trj = Trajectory()
        if ".npz" == filename[-4:]:
            dictionary = dict(np.load(filename, allow_pickle=True))
            trj.copy_dict(dictionary)
        else:
            raise NotImplementedError(f"{filename} format not supported")
        return trj

    @staticmethod
    def from_dict(dictionary):
        trj = Trajectory()
        trj.copy_dict(dictionary)
        return trj

    def to_dict(self):
        data = {}
        for k in self.per_frame_attrs:
            data[k] = getattr(self, k)
        for k in self.metadata_attrs:
            data[k] = getattr(self, k)
        return data

    def copy_dict(self, dictionary):

        nframes = dictionary['positions'].shape[0]

        if 'cells' in dictionary:
            dictionary['cells'] = convert_cell_format(nframes, dictionary['cells'])

        for k in ['positions', 'forces']:
            if k in dictionary:
                dictionary[k] = dictionary[k].reshape([nframes, -1, 3])


        for k in dictionary:
            if k in type(self).per_frame_keys:
                setattr(self, k, np.copy(dictionary[k]))
                self.per_frame_attrs += [k]
            elif k in type(self).metadata_keys:
                setattr(self, k, np.copy(dictionary[k]))
                self.metadata_attrs += [k]
            elif k not in ['per_frame_attrs', 'metadata_attrs']:
                logging.debug(f"undefined attributes {k}, set to metadata")
                setattr(self, k, np.copy(dictionary[k]))
                self.metadata_attrs += [k]
        self.sanity_check()

    def save(self, name: str, format: str = None):

        supported_formats = ['pickle', 'npz', 'xyz', 'poscar']
        format, name = sort_format(supported_formats, format, name)

        if format == 'pickle':
            with open(name, 'wb') as f:
                pickle.dump(self, f)
        elif format == 'npz':
            if '.npz' != name[-4:]:
                name += '.npz'
            data = self.to_dict()
            for k in data:
                try:
                    s = f"write {k}"
                    s += f"{data[k].shape}"
                except:
                    pass
                    # s += f"{data[k]}"
                logging.info(s)
            np.savez(name, **data)
            logging.info(f"! save as {name}")
        elif format == 'xyz':
            for i in range(self.nframes):
                structure = Atoms(cell=self.cells[i].reshape([3, 3]),
                                  symbols=self.species,
                                  positions=self.positions[i].reshape([-1, 3]),
                                  pbc=True)
                write_extxyz(name, structure, append=True)
        elif format == 'poscar':
            for i in range(self.nframes):
                structure = Atoms(cell=self.cells[i].reshape([3, 3]),
                                  symbols=self.species,
                                  positions=self.positions[i].reshape([-1, 3]),
                                  pbc=True)
                write_vasp(f"{i}_{name}", structure, vasp5=True)
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

        self.natom = int(natom)
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
                        if dictionary[k].shape[1] % ori_natom == 0:
                            item = dictionary[k][i].reshape([ori_natom, -1])
                            if idorder is not None:
                                getattr(self, k).append(item[idorder])
                            else:
                                getattr(self, k).append(item)
                        else:
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

            if isinstance(self, Trajectory):
                if type(self) != type(trj):
                    logging.error(f"type {type(self)} != type {type(trj)}")
                    raise RuntimeError("")

            if self.natom != trj.natom and isinstance(self, PaddedTrajectory):
                logging.info(f"adding trajectory with different number of atoms {trj.natom}")
                max_atoms = np.max([self.natom, trj.natom])
                if self.natom < max_atoms:
                    logging.info(f"padd original trj")
                    padded_trj = PaddedTrajectory.from_trajectory(self, max_atoms)
                    self.copy(padded_trj)
                else:
                    logging.info(f"padd the added trj")
                    newtrj = PaddedTrajectory.from_trajectory(trj, max_atoms)
                    trj = newtrj
            elif self.natom != trj.natom:
                logging.info(f"adding trajectory with different number of atoms {trj.natom}")
                raise RuntimeError(f"Trajectory cannot be padded during adding."
                                   " Please initialize as a PaddedTrajectory")

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
        for k in self.per_frame_attrs:
            setattr(trj, k, getattr(self, k)[frame_list])
            trj.per_frame_attrs += [k]
        for k in self.metadata_attrs:
            setattr(trj, k, getattr(self, k))
            trj.metadata_attrs += [k]
        trj.python_list = False
        trj.name = f"{self.name}_{trj.nframes}"
        trj.nframes = len(frame_list)
        trj.sanity_check()

        logging.debug(f"skim {self.nframes} to {trj.nframes}")
        logging.info(f"! generate {repr(trj)}")

        return trj

    def shuffle(self):

        self.convert_to_np()

        frame_list = np.random.permutation(self.nframes)

        for k in self.per_frame_attrs:
            setattr(self, k, getattr(self, k)[frame_list])

    def copy(self, otrj):

        for k in self.per_frame_attrs:
            delattr(self, k)
        self.per_frame_attrs = []
        for k in self.metadata_attrs:
            delattr(self, k)
        self.metadata_attrs = []
        self.empty = True

        for k in otrj.per_frame_attrs:
            setattr(self, k, deepcopy(getattr(otrj, k)))
            self.per_frame_attrs += [k]
        for k in otrj.metadata_attrs:
            setattr(self, k, deepcopy(getattr(otrj, k)))
            self.metadata_attrs += [k]
        self.empty = otrj.empty
        self.sanity_check()


class PaddedTrajectory(Trajectory):

    per_frame_keys = ["natoms", "symbols"] + Trajectory.per_frame_keys

    def __init__(self):
        Trajectory.__init__(self)

    def sanity_check(self):
        Trajectory.sanity_check(self)
        if 'species' in self.metadata_attrs:
            del self.species
            self.metadata_attrs.remove('species')

        assert 'natoms' in self.per_frame_attrs
        assert 'symbols' in self.per_frame_attrs
        self.natoms = np.array(self.natoms, dtype=int)

        if 'natom' not in self.metadata_attrs:
            self.natom = self.positions.shape[1]

    @staticmethod
    def from_trajectory(otrj, max_atom=-1):

        otrj.convert_to_np()
        otrj.sanity_check()

        if max_atom == -1:
            max_atom = otrj.natom

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
                logging.debug(f"{k} before padding {item.shape}")

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
                logging.debug(f"{k} after padding {getattr(trj, k).shape}")

            trj.metadata_attrs = []
            for k in otrj.metadata_attrs:
                if k != 'species':
                    setattr(trj, k, getattr(otrj, k))
                    trj.metadata_attrs += [k]

            trj.name = f"{otrj.name}_padded"
            trj.natom = max_atom

        if isinstance(trj, PaddedTrajectory):
            pad = np.array([['0']*datom]*trj.nframes)
            trj.symbols = np.hstack((otrj.symbols, pad))
            pad = np.zeros((trj.nframes, datom))
        else:
            species = np.hstack([otrj.species, datom*['NA']])
            trj.symbols = np.vstack([species]*trj.nframes)
            trj.natoms = np.ones(trj.nframes)*otrj.natom
            trj.per_frame_attrs += ['symbols']
            trj.per_frame_attrs += ['natoms']

        trj.sanity_check()
        logging.debug(f"! return {repr(trj)}")

        return trj

    @staticmethod
    def from_dict(dictionary):
        trj = PaddedTrajectory()
        trj.copy_dict(dictionary)
        return trj

    @staticmethod
    def from_file(filename):
        trj = PaddedTrajectory()
        if ".npz" == filename[-4:]:
            dictionary = dict(np.load(filename, allow_pickle=True))
            trj.copy_dict(dictionary)
        else:
            raise NotImplementedError(f"{filename} format not supported")
        return trj

    def save(self, name: str, format: str = None):

        supported_formats = ['pickle', 'npz', 'xyz', 'poscar'] # npz

        format, name = sort_format(supported_formats, format, name)

        if format in ['pickle', 'npz']:
            Trajectory.save(self, name, format)
        elif format == 'xyz':
            for i in range(self.nframes):
                natom = self.natoms[i]
                structure = Atoms(cell=self.cells[i].reshape([3, 3]),
                                  symbols=self.symbols[i][:natom],
                                  positions=self.positions[i][:natom].reshape([natom, 3]),
                                  pbc=True)
                write_extxyz(name, structure, append=True)
        elif format == 'poscar':
            for i in range(self.nframes):
                natom = self.natoms[i]
                structure = Atoms(cell=self.cells[i].reshape([3, 3]),
                                  symbols=self.symbols[i][:natom],
                                  positions=self.positions[i][:natom].reshape([natom, 3]),
                                  pbc=True)
                write_vasp(f"{i}{name}", structure, vasp5=True)
        else:
            raise NotImplementedError(f"Output format not supported:"
                                      f" try from {supported_formats}")
