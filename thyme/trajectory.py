import logging
import numpy as np
import pickle

from copy import deepcopy

from ase.atoms import Atoms

from thyme.utils.cell import convert_cell_format
from thyme.utils.save import sort_format
from thyme.utils.atomic_symbols import species_to_order_label
from ._key import *


class Trajectory:

    per_frame_keys = [
        POSITION,
        FORCE,
        TOTAL_ENERGY,
        CELL,
    ]
    metadata_keys = ["dipole_correction", "species", "nelm", "cutoff", "kpoints"]
    fix_fields = []
    switch_keys = ["empty"]
    stat_keys = ["natom", "nframes", "name", "formula"]

    is_padded = False

    def __init__(self):
        """
        dummy init. do nothing
        """

        allkeys = type(self).metadata_keys
        for k in allkeys:
            setattr(self, k, None)

        # unique name that can be used for printing
        self.name = "default"
        self.nframes = 0
        self.natom = 0
        self.formula = ""

        self.empty = True

        self.per_frame_attrs = []
        self.metadata_attrs = []

        self._iter_index = 0

    def __len__(self):
        return self.nframes

    def __repr__(self):
        s = f"{self.name}: {self.nframes} frames with {self.natom} atoms, {self.formula}"
        return s

    def __str__(self):
        """
        string method to list all details or shape of the trajectory
        """

        s = f"{self.name}: {self.nframes} frames with {self.natom} atoms\n("

        for k in self.switch_keys:
            item = getattr(self, k)
            s += f"{k}: {item} "
        s += ")\n"

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

    def __getitem__(self, key):
        if key in self.per_frame_attrs or key in self.metadata_attrs:
            return getattr(self, key, None)
        if isinstance(key, int):
            return self.get_frame(key)

    def __iter__(self):
        return self

    def __next__(self):

        self._iter_index = getattr(self, "_iter_index", 0)

        n_attrs = len(self.per_frame_attrs)
        if self._iter_index >= n_attrs:
            raise StopIteration
        key = self.per_frame_attrs[self._iter_index]
        self._iter_index += 1
        return getattr(self, key, None)

    def get_frame(self, idx):
        if idx >= self.nframes:
            raise ValueError(f"{idx} is larger than the total length {self.nframes}")
        frame = {key: getattr(self, key)[idx] for key in self.per_frame_attrs}
        for key in self.fix_fields:
            frame[key] = getattr(self, key)
        return frame

    @property
    def keys(self):
        return self.per_frame_attrs

    def sanity_check(self):

        if self.empty and (self.natom > 0 or self.nframes > 0):
            raise RuntimeError("atom error")

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
                logging.error(
                    f"numbers of frames are inconsistent: {frames} and nframes = {self.nframes}"
                )
                raise RuntimeError(f"Data inconsistent")

            self.per_frame_attrs = list(set(self.per_frame_attrs))

            if self.cell.shape[1] != 3 and self.cell.shape[2] != 3:
                logging.error(
                    f"cell should be 3x3 : {frames} and nframes = {self.nframes}"
                )
                raise RuntimeError(f"Data inconsistent")

        self.per_frame_attrs = list(set(self.per_frame_attrs))
        self.metadata_attrs = list(set(self.metadata_attrs))

        if "natom" not in self.metadata_attrs:
            logging.error("natom is not in metadata_attrs")
            raise RuntimeError(f"Data inconsistent")

    def add_per_frame_attr(self, name, item):

        if len(item) != self.nframes:
            logging.error(
                f"Error: {repr(item)}'s length {len(item)} does not match {self.nframes}"
            )
            raise RuntimeError

        if name in self.per_frame_attrs:
            logging.debug(f"overwriting per_frame attr {name}")
        else:
            self.per_frame_attrs += [name]

        setattr(self, name, item)
        logging.debug(f"add {name} to the system")

        # always put POSITION as the first attribute
        if POSITION in self.per_frame_attrs:
            idx = self.per_frame_attrs.index(POSITION)
            if idx != 0:
                temp = self.per_frame_attrs[0]
                self.per_frame_attrs[0] = POSITION
                self.per_frame_attrs[idx] = temp

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
    def from_file(filename, per_frame_attrs=None, mapping=None):
        trj = Trajectory()
        if ".npz" == filename[-4:]:
            dictionary = dict(np.load(filename, allow_pickle=True))
            trj.copy_dict(dictionary, per_frame_attrs, mapping=mapping)
        elif ".pickle" == filename[-7:]:
            with open(filename, "rb") as fin:
                trj = pickle.load(fin)
        else:
            raise NotImplementedError(f"{filename} format not supported")
        return trj

    @staticmethod
    def from_dict(dictionary, per_frame_attrs=None, mapping=None):
        trj = Trajectory()
        trj.copy_dict(dictionary, per_frame_attrs, mapping=mapping)
        return trj

    def to_dict(self):
        data = {}
        for k in self.per_frame_attrs:
            data[k] = getattr(self, k)
        for k in self.metadata_attrs:
            data[k] = getattr(self, k)
        return data

    def copy_dict(self, dictionary, per_frame_attrs=None, mapping=None):
        """

        requirement

        positions: nframes x ?

        optional:
            cells
            forces

        species, or symbols

        """

        self.clean_containers()

        if mapping is not None:
            for new_name, original_name in mapping.items():
                dictionary[new_name] = dictionary.pop(original_name)

        nframes = dictionary[POSITION].shape[0]
        self.nframes = nframes

        if CELL in dictionary:
            dictionary[CELL] = convert_cell_format(nframes, dictionary[CELL])

        for k in ["positions", FORCE]:
            if k in dictionary:
                dictionary[k] = dictionary[k].reshape([nframes, -1, 3])

        if per_frame_attrs is not None:
            for k in per_frame_attrs:
                self.per_frame_keys += [k]

        if "per_frame_attrs" in dictionary:
            for k in dictionary["per_frame_attrs"]:
                self.per_frame_keys += [k]
        if "metadata_attrs" in dictionary:
            for k in dictionary["metadata_attrs"]:
                self.metadata_keys += [k]

        for k in dictionary:

            if k in type(self).stat_keys:

                setattr(self, k, dictionary[k])

            elif k in type(self).per_frame_keys:

                self.add_per_frame_attr(k, deepcopy(dictionary[k]))

            elif k in type(self).metadata_keys:

                setattr(self, k, deepcopy(dictionary[k]))
                self.metadata_attrs += [k]

            elif k not in ["per_frame_attrs", "metadata_attrs"]:

                logging.debug(f"undefined attributes {k}, set to metadata")
                try:
                    dim0 = dictionary[k].shape[0]
                except:
                    dim0 = -1
                if dim0 == nframes:
                    self.add_per_frame_attr(k, deepcopy(dictionary[k]))
                else:
                    self.add_metadata(k, deepcopy(dictionary[k]))

        self.empty = False
        self.sanity_check()

    def copy_metadata(self, trj, exception):

        for k in set(trj.metadata_attrs) - set(exception):
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
                    logging.info(f"key {k} are not the same in the two objects")
                    logging.info(f"        {item} {ori_item}")

    def save(self, name: str, format: str = None):

        supported_formats = ["pickle", "npz"]
        format, name = sort_format(supported_formats, format, name)

        if format == "pickle":
            with open(name, "wb") as f:
                pickle.dump(self, f)
        elif format == "npz":
            if ".npz" != name[-4:]:
                name += ".npz"
            data = self.to_dict()
            logging.info(f"saving {self}")
            np.savez(name, **data)
            logging.info(f"! save as {name}")
        else:
            raise NotImplementedError(
                f"Output format not supported:" f" try from {supported_formats}"
            )

    def clean_containers(self):

        for k in self.per_frame_attrs:
            delattr(self, k)
        self.per_frame_attrs = []

        for k in self.metadata_attrs:
            delattr(self, k)

        self.nframes = 0
        self.natom = 0
        self.species = []
        self.empty = True
        self.name = ""

        self.per_frame_attrs = []
        self.metadata_attrs = [
            "nframes",
            "name",
            "empty",
            "species",
            "natom",
        ]

    def add_containers(self, natom: int = 0, per_frame_attrs: list = []):
        """
        initialize all attributes with empty numpy array

        attributes: only per_frame_attrs needs to be listed
        """

        if not self.empty:
            return

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
                    f"adding trajectory with different number of atoms {trj.natom}"
                )
                raise RuntimeError(
                    f"Trajectory cannot be padded during adding."
                    " Please initialize as a PaddedTrajectory"
                )
            elif self.natom != trj.natom:

                max_atoms = np.max([self.natom, trj.natom])

                if self.natom < max_atoms:
                    self.increase_maxatom(max_atom)
                else:
                    padded_trj = PaddedTrajectory.from_trajectory(trj, max_atoms)
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

            self.copy_metadata(trj, exception=["name", "nframes", "natom", "filenames"])

            self.nframes += trj.nframes

    def reshape(self):

        if len(self.position.shape) == 2:
            self.position = self.position.reshape([self.nframes, self.natoms, 3])
        if len(self.force.shape) == 2:
            self.force = self.force.reshape([self.nframes, self.natoms, 3])
        if len(self.cell.shape) == 2:
            self.cell = self.cell.reshape([self.nframes, 3, 3])

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

        if len(self.position.shape) == 3:
            self.position = self.position.reshape([self.nframes, -1])
        if len(self.force.shape) == 3:
            self.force = self.force.reshape([self.nframes, -1])
        if len(self.cell.shape) == 3:
            self.cell = self.cell.reshape([self.nframes, 9])

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

        natom = self.position.shape[1]

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

            natoms = np.ones(otrj.nframes) * otrj.natom

            symbols = np.hstack([otrj.species] * self.nframes).reshape(
                [self.nframes, -1]
            )
            self.add_per_frame_attr("symbols", symbols)
            self.add_per_frame_attr("natoms", natoms)

        elif otrj.is_padded and not self.is_padded:

            del self.symbols
            del self.natoms
            self.per_frame_attrs.remove("symbols")
            self.per_frame_attrs.remove("natoms")

            if len(set(otrj.natoms)) != 1:
                raise RuntimeError(
                    "cannot convert a padded_trj to trj with different length"
                )

            species = otrj.symbols[0]
            self.species = species
            self.natom = len(species)

            reorder = False
            for i in range(otrj.nframes):
                if not all(species == otrj.symbols[i][: self.natom]):
                    c1 = dict(Counter(species))
                    c2 = dict(Counter(otrj.symbols[i][: self.natom]))
                    if set(list(c1.keys())) != set(list(c2.keys())):
                        raise RuntimeError(
                            "cannot convert a padded_trj to trj with "
                            "different types of component"
                        )
                    for ele in c1:
                        if c1[ele] != c2[ele]:
                            raise RuntimeError(
                                "cannot convert a padded_trj to trj with "
                                "different numbers of component for each type"
                            )
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
            #     self.natom = self.position.shape[1]

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

        if "species" in self.metadata_attrs:
            del self.species
            self.metadata_attrs.remove("species")

        assert "natoms" in self.per_frame_attrs
        assert "symbols" in self.per_frame_attrs
        self.natoms = np.array(self.natoms, dtype=int)

    @staticmethod
    def from_trajectory(otrj, max_atom=-1):
        trj = PaddedTrajectory()
        trj.copy(otrj, max_atom)
        return trj

    @staticmethod
    def from_dict(dictionary, per_frame_attrs=None, mapping=None):
        trj = PaddedTrajectory()
        trj.copy_dict(dictionary, per_frame_attrs, mapping=mapping)
        return trj

    @staticmethod
    def from_file(filename, per_frame_attrs=None, mapping=None):
        trj = PaddedTrajectory()
        if ".npz" == filename[-4:]:
            dictionary = dict(np.load(filename, allow_pickle=True))
            trj.copy_dict(dictionary, per_frame_attrs, mapping=mapping)
        else:
            raise NotImplementedError(f"{filename} format not supported")
        return trj

    def save(self, name: str, format: str = None):

        supported_formats = ["pickle", "npz"]

        format, name = sort_format(supported_formats, format, name)

        if format in ["pickle", "npz"]:
            Trajectory.save(self, name, format)
        else:
            raise NotImplementedError(
                f"Output format not supported:" f" try from {supported_formats}"
            )

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
                    self.increase_maxatom(max_atoms)
                else:
                    padded_trj = PaddedTrajectory.from_trajectory(trj, max_atoms)
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
                trj, exception=["name", "nframes", "species", "filenames"]
            )
            self.nframes += trj.nframes

    def increase_maxatom(self, max_atom):

        if max_atom == -1:
            return

        logging.info(f"increase {self.name} from size {self.natom} to size {max_atom}")

        datom = max_atom - self.natom
        if datom < 0:
            raise RuntimeError("wrong max atom is set")

        pad = np.array([["0"] * datom] * self.nframes)
        self.symbols = np.hstack((self.symbols, pad))

        for k in self.per_frame_attrs:
            if k == "symbols":
                continue

            item = getattr(self, k)
            dim = len(item.shape)

            if dim >= 2:

                if item.shape[1] == self.natom:
                    new_shape = list(item.shape)
                    new_shape[1] = max_atom - new_shape[1]
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
        if delete_elements != set(["0"]):
            logging.warning("Warning some {delete_elements} atoms will be removed")

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

        natom = self.position.shape[1]

        if natom != self.natom:
            logging.info(f"skim {self.natom} atoms to {natom} atoms")

        self.natom = natom
        self.sanity_check()
