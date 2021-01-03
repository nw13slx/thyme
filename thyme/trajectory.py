import logging
import numpy as np
import pickle

from copy import deepcopy

from ase.atoms import Atoms

from thyme.utils.cell import convert_cell_format
from thyme.utils.save import sort_format


class Trajectory():

    per_frame_keys = ["positions", "forces", "energies",
                      "cells", 'pe', 'label', 'labels']
    metadata_keys = ["dipole_correction", "species",
                     "nelm", "nframes", "cutoff",
                     "natom", "kpoints", "empty",
                     "python_list", "name"]
    is_padded = False

    def __init__(self):
        """
        dummy init. do nothing
        """

        allkeys = type(self).per_frame_keys + type(self).metadata_keys
        for k in allkeys:
            setattr(self, k, None)

        self.nframes = 0
        self.natom = 0
        self.species = []
        self.python_list = False
        self.empty = True
        self.name = ""

        self.per_frame_attrs = []
        self.metadata_attrs = ['nframes', 'name',
                               'python_list', 'empty', 'species', 'natom']

    def __repr__(self):
        s = f"{self.name}: {self.nframes} frames with {self.natom} atoms"
        return s

    def __str__(self):
        s = f"{self.name}: {self.nframes} frames with {self.natom} atoms\n"
        for k in self.per_frame_attrs:
            item = getattr(self, k)
            s += f"  {k} {item.shape}\n"
        s += "metadata:\n"
        for k in self.metadata_attrs:
            item = getattr(self, k)
            if isinstance(item, np.ndarray):
                s += f"  {k} shape {item.shape}\n"
            elif isinstance(item, np.ndarray):
                s += f"  {k} len {len(item)}\n"
            else:
                s += f"  {k} value {item}\n"
        return s

    def sanity_check(self):

        if len(self.per_frame_attrs) != 0:

            self.empty = False

            frames = {}
            for k in self.per_frame_attrs:
                frames[k] = len(getattr(self, k))
            frames_values = set(list(frames.values()))

            if len(frames_values) > 1:
                logging.error(f"numbers of frames are inconsistent: {frames}")
                raise RuntimeError(f"Data inconsistent")

            if self.nframes != list(frames_values)[0]:
                logging.error(f"numbers of frames are inconsistent: {frames} and nframes = {self.nframes}")
                raise RuntimeError(f"Data inconsistent")

            self.per_frame_attrs = list(set(self.per_frame_attrs))

            if self.cells.shape[1] != 3 and self.cells.shape[2] != 3:
                logging.error(f"cell should be 3x3 : {frames} and nframes = {self.nframes}")
                raise RuntimeError(f"Data inconsistent")

        self.metadata_attrs = list(set(self.metadata_attrs))

        if 'natom' not in self.metadata_attrs:
            logging.error('natom is not in metadata_attrs')
            raise RuntimeError(f"Data inconsistent")

    def add_per_frame_attr(self, name, item):

        if len(item) != self.nframes:
            logging.error(f"Error: {repr(item)}\'s length {len(item)} does not match {self.nframes}")
            raise RuntimeError

        if name in self.per_frame_attrs:
            logging.debug(f"overwriting per_frame attr {name}")
        else:
            self.per_frame_attrs += [name]

        setattr(self, name, item)
        logging.debug(f"add {name} to the system")

    def add_metadata(self, name, item):

        if name in self.metadata_attrs:
            logging.debug(f"overwriting metadata {name}")

        self.metadata_attrs += [name]
        setattr(self, name, item)
        logging.debug(f"add {name} to the system")

    def filter_frames(self, accept_id=None):

        if accept_id is None:
            return

        for k in self.per_frame_attrs:
            new_mat = getattr(self, k)
            new_mat = getattr(self, k)[accept_id]
            setattr(self, k, new_mat)
        self.nframes = len(accept_id)

    @staticmethod
    def from_file(filename, per_frame_attrs=None):
        trj = Trajectory()
        if ".npz" == filename[-4:]:
            dictionary = dict(np.load(filename, allow_pickle=True))
            trj.copy_dict(dictionary, per_frame_attrs)
        elif '.pickle' == filename[-7:]:
            with open(filename, "rb") as fin:
                trj = pickle.load(fin)
        else:
            raise NotImplementedError(f"{filename} format not supported")
        return trj

    @staticmethod
    def from_dict(dictionary, per_frame_attrs=None):
        trj = Trajectory()
        trj.copy_dict(dictionary, per_frame_attrs)
        return trj

    def to_dict(self):
        data = {}
        for k in self.per_frame_attrs:
            data[k] = getattr(self, k)
        for k in self.metadata_attrs:
            data[k] = getattr(self, k)
        return data

    def copy_dict(self, dictionary, per_frame_attrs=None):
        """

        requirement

        positions: nframes x ?

        optional:
            cells
            forces

        species, or symbols

        """

        self.clean_containers()

        nframes = dictionary['positions'].shape[0]
        self.nframes = nframes

        if 'cells' in dictionary:
            dictionary['cells'] = convert_cell_format(
                nframes, dictionary['cells'])

        for k in ['positions', 'forces']:
            if k in dictionary:
                dictionary[k] = dictionary[k].reshape([nframes, -1, 3])

        if per_frame_attrs is not None:
            for k in per_frame_attrs:
                self.per_frame_keys += [k]

        if 'per_frame_attrs' in dictionary:
            for k in dictionary['per_frame_attrs']:
                self.per_frame_keys += [k]
        if 'metadata_attrs' in dictionary:
            for k in dictionary['metadata_attrs']:
                self.metadata_keys += [k]

        for k in dictionary:

            if k in type(self).per_frame_keys:

                self.add_per_frame_attr(k, deepcopy(dictionary[k]))

            elif k in type(self).metadata_keys:

                setattr(self, k, deepcopy(dictionary[k]))
                self.metadata_attrs += [k]

            elif k not in ['per_frame_attrs', 'metadata_attrs']:

                logging.debug(f"undefined attributes {k}, set to metadata")
                try:
                    dim0 = dictionary[k].shape[0]
                except:
                    dim0 = -1
                if dim0 == nframes:
                    self.add_per_frame_attr(k, deepcopy(dictionary[k]))
                else:
                    self.add_metadata(k, deepcopy(dictionary[k]))

        self.sanity_check()

    def copy_metadata(self, trj, exception):

        for k in set(trj.metadata_attrs)-set(exception):
            item = getattr(trj, k, None)
            ori_item = getattr(self, k, None)
            if ori_item is None and item is not None:
                setattr(self, k, item)
                if k not in self.metadata_attrs:
                    self.metadata_attrs += [k]
            else:

                equal = False
                try:
                    if ori_item == item:
                        equal = True
                except:
                    pass

                try:
                    if (ori_item == item).all():
                        equal = True
                except:
                    pass

                try:
                    if np.equal(ori_item, item).all():
                        equal = True
                except:
                    pass

                if not equal and item is None:
                    ori_item = getattr(self, k, None)
                    logging.info(
                        f"key {k} are not the same in the two objects")
                    logging.info(f"        {item} {ori_item}")

    def save(self, name: str, format: str = None):

        supported_formats = ['pickle', 'npz']
        format, name = sort_format(supported_formats, format, name)

        if format == 'pickle':
            with open(name, 'wb') as f:
                pickle.dump(self, f)
        elif format == 'npz':
            if '.npz' != name[-4:]:
                name += '.npz'
            data = self.to_dict()
            logging.info(f"saving {self}")
            np.savez(name, **data)
            logging.info(f"! save as {name}")
        else:
            raise NotImplementedError(f"Output format not supported:"
                                      f" try from {supported_formats}")

    def clean_containers(self):

        for k in self.per_frame_attrs:
            delattr(self, k)
        self.per_frame_attrs = []

        for k in self.metadata_attrs:
            delattr(self, k)

        self.nframes = 0
        self.natom = 0
        self.species = []
        self.python_list = False
        self.empty = True
        self.name = ""

        self.per_frame_attrs = []
        self.metadata_attrs = ['nframes', 'name',
                               'python_list', 'empty', 'species', 'natom']

    def add_containers(self, natom: int = 0,
                       per_frame_attrs: list = []):
        """
        initialize all attributes with empty list (python_list = True)
        or numpy array (python_list = False)

        attributes: only per_frame_attrs needs to be listed
        """

        if not self.empty:
            return

        if self.python_list:
            item = []
        else:
            item = None

        for k in per_frame_attrs:
            self.add_per_frame_attr(k, item)

        self.natom = int(natom)


    @staticmethod
    def from_padded_trajectory(otrj):

        trj = Trajectory()
        trj.copy(otrj)
        return trj

    def add_trj(self, trj):
        """
        add all frames from another trajectory instance
        """

        if trj.nframes <= 0:
            return

        if self.empty:
            self.copy(trj)
            self.convert_to_np()
        else:

            self.convert_to_np()

            if trj.is_padded and not self.is_padded:
                logging.error(f"type {type(self)} != type {type(trj)}")
                raise RuntimeError("")

            if self.natom != trj.natom and not self.is_padded:
                logging.info(
                    f"adding trajectory with different number of atoms {trj.natom}")
                raise RuntimeError(f"Trajectory cannot be padded during adding."
                                   " Please initialize as a PaddedTrajectory")
            elif self.natom != trj.natom:

                max_atoms = np.max([self.natom, trj.natom])

                if self.natom < max_atoms:
                    self.increase_maxatom(max_atom)
                else:
                    padded_trj = PaddedTrajectory.from_trajectory(
                        trj, max_atoms)
                    trj = padded_trj

                if not trj.is_padded:
                    trj = PaddedTrajectory.from_trajectory(trj, trj.natom)

            for k in self.per_frame_attrs:
                item = getattr(trj, k)
                ori_item = getattr(self, k)
                if len(item.shape) == 1:
                    setattr(self, k, np.hstack((ori_item, item)))
                else:
                    setattr(self, k, np.vstack((ori_item, item)))
                ori_item = getattr(self, k)

            self.copy_metadata(
                trj, exception=['name', 'nframes', 'natom', 'filenames'])

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

        self.sanity_check()

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
            self.positions = self.positions.reshape(
                [self.nframes, self.natoms, 3])
        if len(self.forces.shape) == 2:
            self.forces = self.forces.reshape([self.nframes, self.natoms, 3])
        if len(self.cells.shape) == 2:
            self.cells = self.cells.reshape([self.nframes, 3, 3])

        if self.natom > 2:
            for k in self.per_frame_attrs:
                item = getattr(self, k)
                try:
                    if item.shape[1] % self.natom == 0:
                        item = item.reshape([self.nframes, self.natoms, -1])
                        setattr(self, k, item)
                except:
                    pass


    def flatten(self):

        if len(self.positions.shape) == 3:
            self.positions = self.positions.reshape([self.nframes, -1])
        if len(self.forces.shape) == 3:
            self.forces = self.forces.reshape([self.nframes, -1])
        if len(self.cells.shape) == 3:
            self.cells = self.cells.reshape([self.nframes, 9])

    def skim(self, frame_list):

        self.convert_to_np()

        if self.is_padded:
            trj = PaddedTrajectory()
        else:
            trj = Trajectory()
        trj.copy(self)

        trj.filter_frames(frame_list)
        trj.name = f"{self.name}_{trj.nframes}"
        trj.nframes = len(frame_list)
        trj.sanity_check()

        return trj

    def reorder(self, orders):

        if len(orders) != self.natom:
            logging.error(f"{len(orders)} order vs {self.natom} atoms")
            raise RuntimeError()

        for k in self.per_frame_attrs:
            ori_item = getattr(self, k)
            if len(ori_item.shape) > 1:
                if ori_item.shape[1] == self.natom:
                    item = np.swapaxes(ori_item, 0, 1)
                    item = item[orders]
                    item = np.swapaxes(item, 0, 1)
                    setattr(self, k, item)

        self.species = np.array(self.species)[orders]

        natom = self.positions.shape[1]

        if natom != self.natom:
            logging.info(f"skim {self.natom} atoms to {natom} atoms")

        self.natom = natom
        self.sanity_check()

    def shuffle(self):

        self.convert_to_np()

        frame_list = np.random.permutation(self.nframes)

        for k in self.per_frame_attrs:
            setattr(self, k, getattr(self, k)[frame_list])

    def copy(self, otrj, max_atom=-1):

        self.clean_containers()

        self.nframes = otrj.nframes

        for k in otrj.per_frame_attrs:
            self.add_per_frame_attr(k, deepcopy(getattr(otrj, k)))

        for k in otrj.metadata_attrs:
            self.add_metadata(k, deepcopy(getattr(otrj, k)))

        self.empty = otrj.empty

        if self.is_padded and not otrj.is_padded:

            self.name = f"{otrj.name}_padded"

            natoms = np.ones(otrj.nframes)*otrj.natom

            symbols = np.hstack(
                [otrj.species]*self.nframes).reshape([self.nframes, -1])
            self.add_per_frame_attr('symbols', symbols)
            self.add_per_frame_attr('natoms', natoms)

        elif otrj.is_padded and not self.is_padded:


            del self.symbols
            del self.natoms
            self.per_frame_attrs.remove('symbols')
            self.per_frame_attrs.remove('natoms')

            if len(set(otrj.natoms)) != 1:
                raise RuntimeError(
                    "cannot convert a padded_trj to trj with different length")

            species = otrj.symbols[0]
            self.species = species
            self.natom = len(species)

            reorder = False
            for i in range(otrj.nframes):
                if not all(species == otrj.symbols[i][:self.natom]):
                    c1 = dict(Counter(species))
                    c2 = dict(Counter(otrj.symbols[i][:self.natom]))
                    if set(list(c1.keys())) != set(list(c2.keys())):
                        raise RuntimeError(
                            "cannot convert a padded_trj to trj with "
                            "different types of component")
                    for ele in c1:
                        if c1[ele] != c2[ele]:
                            raise RuntimeError(
                                "cannot convert a padded_trj to trj with "
                                "different numbers of component for each type")
                    reorder = True

            # TO DO: auto truncation
            # self.natom = otrj.natoms[0]
            # for k in otrj.per_frame_attrs:
            #     item = getattr(self, k)
            #     item = item[:, :self.natom]
            #     setattr(self, k, item)


            # TO DO, auto reorder
            # if reorder:

            #     order, label = species_to_order_label(otrj.symbols[0][:self.natom])

            #     self.species = otrj.symbols[0][order]
            #     self.natom = self.positions.shape[1]

            #     for i in range(otrj.nframes):
            #         order, label = species_to_order_label(otrj.symbols[i][:self.natom])
            #         for k in self.per_frame_attrs:
            #             item = getattr(self, k)
            #             try:
            #                 if item.shape[1] == self.natom:
            #                     item[i] = item[i][order]
            #             except:
            #                 pass

        if self.is_padded:
            self.increase_maxatom(max_atom)
        self.sanity_check()


class PaddedTrajectory(Trajectory):
    """
    Trajectory type that store configurations from different simulations with different compositions.
    It is almost the same as the parent class, Trajectory, except that it has natoms and symbols attribute,
    which stores the number and species of atoms for each config.
    The element symbol '0' indicate
    """

    per_frame_keys = ["natoms", "symbols"] + Trajectory.per_frame_keys
    is_padded = True

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

    @staticmethod
    def from_trajectory(otrj, max_atom=-1):
        trj = PaddedTrajectory()
        trj.copy(otrj, max_atom)
        return trj

    @staticmethod
    def from_dict(dictionary, per_frame_attrs=None):
        trj = PaddedTrajectory()
        trj.copy_dict(dictionary, per_frame_attrs)
        return trj

    @staticmethod
    def from_file(filename, per_frame_attrs=None):
        trj = PaddedTrajectory()
        if ".npz" == filename[-4:]:
            dictionary = dict(np.load(filename, allow_pickle=True))
            trj.copy_dict(dictionary, per_frame_attrs)
        else:
            raise NotImplementedError(f"{filename} format not supported")
        return trj

    def save(self, name: str, format: str = None):

        supported_formats = ['pickle', 'npz']

        format, name = sort_format(supported_formats, format, name)

        if format in ['pickle', 'npz']:
            Trajectory.save(self, name, format)
        else:
            raise NotImplementedError(f"Output format not supported:"
                                      f" try from {supported_formats}")


    def add_trj(self, trj):
        """
        add all frames from another trajectory instance
        """

        if trj.nframes <= 0:
            return

        if self.empty:
            self.copy(trj)
            self.convert_to_np()
        else:

            self.convert_to_np()


            if self.natom != trj.natom:
                max_atoms = np.max([self.natom, trj.natom])

                if self.natom < max_atoms:
                    self.increase_maxatom(max_atom)
                else:
                    padded_trj = PaddedTrajectory.from_trajectory(
                        trj, max_atoms)
                    trj = padded_trj

            if not trj.is_padded:
                trj = PaddedTrajectory.from_trajectory(trj, trj.natom)

            for k in self.per_frame_attrs:
                ori_item = getattr(self, k)
                item = getattr(trj, k)

                logging.debug(f"merging {k} {ori_item.shape} {item.shape}")
                if len(item.shape) == 1:
                    setattr(self, k, np.hstack((ori_item, item)))
                else:
                    setattr(self, k, np.vstack((ori_item, item)))
                ori_item = getattr(self, k)

            self.copy_metadata(
                trj, exception=['name', 'nframes', 'species', 'filenames'])
            self.nframes += trj.nframes

    def increase_maxatom(self, max_atom):

        if max_atom == -1:
            return

        logging.info(f"increase {self.name} from size {self.natom} to size {max_atom}")

        datom = max_atom-self.natom
        if datom < 0:
            raise RuntimeError("wrong max atom is set")

        pad = np.array([['0']*datom]*self.nframes)
        self.symbols = np.hstack((self.symbols, pad))

        for k in self.per_frame_attrs:
            if k == 'symbols':
                continue

            item = getattr(self, k)
            dim = len(item.shape)

            if dim >= 2:

                if item.shape[1] == self.natom:
                    new_shape = list(item.shape)
                    new_shape[1] = max_atom-new_shape[1]
                    pad = np.zeros(new_shape)
                    new_item = np.hstack([item, pad])
                    setattr(self, k, new_item)

        self.natom = max_atom
        self.sanity_check()

    def decrease_maxatom(self, max_atom):

        if max_atom == -1:
            return

        datom = self.natom - max_atom
        if datom < 0:
            raise RuntimeError("wrong max atom is set")

        delete_elements = set(self.symbols[:, max_atom:])
        if delete_elements != set(['0']):
            logging.warning('Warning some {delete_elements} atoms will be removed')

        for k in self.per_frame_attrs:

            item = getattr(self, k)
            dim = len(item.shape)

            if dim >= 2:
                if item.shape[1] == self.natom:
                    item = item[:, :max_atom]
                    setattr(trj, k, item)

        self.natom = max_atom
        self.sanity_check()


    def to_Trajectory(self):
        trj = Trajectory()
        trj.copy(self)
        return trj

    def reorder(self, orders):

        if len(orders) != self.nframes:
            logging.error(f"{len(orders)} orders vs {self.nframes} frames")
            raise RuntimeError()

        for k in self.per_frame_attrs:
            ori_item = getattr(self, k)
            if len(ori_item.shape) > 1:
                if ori_item.shape[1] == self.natom:
                    item = []
                    for iconfig in range(self.nframes):
                        order = orders[iconfig]
                        if order is not None:
                            item += [[ori_item[iconfig, order]]]
                        else:
                            item += [[ori_item[iconfig, :]]]
                    item = np.vstack(item)
                    setattr(self, k, item)

        natom = self.positions.shape[1]

        if natom != self.natom:
            logging.info(f"skim {self.natom} atoms to {natom} atoms")

        self.natom = natom
        self.sanity_check()
