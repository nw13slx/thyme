import logging
import numpy as np
import pickle

from copy import deepcopy

from ase.atoms import Atoms

from thyme.utils.cell import convert_cell_format
from thyme.utils.savenload import save_file, load_file
from thyme.utils.atomic_symbols import species_to_order_label
from ._key import *


class Trajectory(object):

    default_per_frame_keys = [
        POSITION,
        FORCE,
        TOTAL_ENERGY,
        CELL,
    ]
    default_metadata_keys = [
        "dipole_correction",
        "species",
        "nelm",
        "cutoff",
        "kpoints",
    ]
    stat_keys = [
        NATOMS,
        SPECIES,
        "nframes",
        "name",
        "formula",
        PER_FRAME_ATTRS,
        METADATA_ATTRS,
        "fixed_attrs",
        "filenames",
    ]

    def __init__(self):
        """
        dummy init. do nothing
        """
        self._items = dict()

        # unique name that can be used for printing
        self.name = "default"
        self.nframes = 0
        self.natoms = 0
        self.formula = ""

        self.per_frame_attrs = []
        self.metadata_attrs = []
        self.fixed_attrs = []

        self._iter_index = 0

    def __len__(self):
        return self.nframes

    def __repr__(self):
        s = f"{self.name}: {self.nframes} frames with {self.natoms} atoms, {self.formula}"
        return s

    def __str__(self):
        """
        string method to list all details or shape of the trajectory
        """

        s = repr(self)

        for k in self.per_frame_attrs:
            item = getattr(self, k)
            s += f"\n  {k} {item.shape}"
        s += "metadata:\n"
        for k in self.metadata_attrs:
            item = getattr(self, k)
            if isinstance(item, np.ndarray):
                s += f"\n  {k} shape {item.shape}"
            elif isinstance(item, np.ndarray):
                s += f"\n  {k} len {len(item)}"
            else:
                s += f"\n  {k} value {item}"
        return s

    @property
    def keys(self):
        return (
            self.per_frame_attrs
            + self.metadata_attrs
            + self.fixed_attrs
            + self.stat_keys
        )

    def __getitem__(self, key):
        if key in self.per_frame_attrs or key in self.metadata_attrs:
            return getattr(self, key, None)
        if isinstance(key, int):
            return self.get_frame(key)

    def __iter__(self):
        return self

    def __next__(self):

        self._iter_index = getattr(self, "_iter_index", 0)

        if self._iter_index >= self.nframes:
            raise StopIteration
        self._iter_index += 1
        return self.get_frame(self._iter_index - 1)

    def get_frame(self, idx, keys=None):
        if idx >= self.nframes:
            raise ValueError(f"{idx} is larger than the total length {self.nframes}")
        frame = {NATOMS: self.natoms, SPECIES: self.species}
        if keys is None:
            frame.update({key: getattr(self, key)[idx] for key in self.per_frame_attrs})
            frame.update({key: getattr(self, key) for key in self.fixed_attrs})
        else:
            frame.update(
                {
                    key: getattr(self, key)[idx]
                    for key in self.per_frame_attrs
                    if key in keys
                }
            )
            frame.update(
                {key: getattr(self, key) for key in self.fixed_attrs if key in keys}
            )
        return frame

    def add_frames(self, dictionary):
        find_key = [(key in dictionary) for key in self.per_frame_attrs]
        if not all(find_key):
            raise RuntimeError("key missing")

        match_fields = [
            (repr(dictionary[key]) == getattr(self.key))
            for key in self.fixed_attrs
            if key in dictionary
        ]
        if not all(match_fields):
            raise RuntimeError("fixed fields are not consistent missing")

        for key in self.per_frame_attrs:
            mat = np.vstack((getattr(self, key), dictionary[key]))
            setattr(self, key, mat)

        self.nframes += dictionary[POSITION].shape[0]

    def get_attr(self, key):
        if key in self.per_frame_attrs:
            return getattr(self, key)
        else:
            return np.array([getattr(self, key)] * self.nframes)

    def pop(self, key, fail=None):
        item = self.get_attr(key)
        for name_list in [self.per_frame_attrs, self.fixed_attrs, self.metadata_attrs]:
            if key in name_list:
                name_list.remove(key)
        delattr(self, key)
        return item

    def add_frames(self, dictionary):
        find_key = [(key in dictionary) for key in self.per_frame_attrs]
        if not all(find_key):
            raise RuntimeError("key missing")

        match_fields = [
            (repr(dictionary[key]) == getattr(self.key))
            for key in self.fixed_attrs
            if key in dictionary
        ]
        if not all(match_fields):
            raise RuntimeError("fixed fields are not consistent missing")

        for key in self.per_frame_attrs:
            mat = np.append(getattr(self, key), dictionary[key], axis=0)
            setattr(self, key, mat)

        self.nframes += dictionary[POSITION].shape[0]

    def sanity_check(self):

        for k in self.stat_keys:
            if not hasattr(self, k):
                setattr(self, k, 0)

        if self.nframes < 0:
            raise RuntimeError("nframes should be non-negative int")

        if len(self.per_frame_attrs) != 0:

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

            if len(self.per_frame_attrs) > len(list(set(self.per_frame_attrs))):
                raise ValueError(f"repeated keys in self.per_frame_attrs")

            # always put POSITION as the first attribute
            if POSITION not in self.per_frame_attrs:
                raise ValueError(POSITION + " has to be defined")

            idx = self.per_frame_attrs.index(POSITION)
            if idx != 0:
                temp = self.per_frame_attrs[0]
                self.per_frame_attrs[0] = POSITION
                self.per_frame_attrs[idx] = temp

            if (
                self.position.shape[1] != self.natoms
                or len(self.species) != self.natoms
            ):
                if self.position.shape[1] == len(self.species):
                    self.natoms = self.position.shape[1]
                else:
                    raise ValueError("Natoms cannot be defined")

        if len(self.metadata_attrs) > len(list(set(self.metadata_attrs))):
            raise ValueError(f"repeated keys in self.metadata_attrs")
        if len(self.fixed_attrs) > len(list(set(self.fixed_attrs))):
            raise ValueError(f"repeated keys in self.fix_attrs")

        if len(set(self.fixed_attrs).intersection(set(self.per_frame_attrs))) > 0:
            raise ValueError(
                f"fix attr: {self.fixed_attrs} and per frame attr: {self.per_frame_attrs} has overlap"
            )

    def add_field(self, list_name, name, item):

        if list_name == PER_FRAME_ATTRS:
            if not isinstance(item, np.ndarray):
                raise TypeError(f"{name} value should be np.ndarray type")

            if len(item) != self.nframes and len(self.per_frame_attrs) > 0:
                logging.error(
                    f"Error: {repr(item)}'s length {len(item)} does not match {self.nframes}"
                )
                raise RuntimeError
            elif len(self.per_frame_attrs) == 0:
                self.nframes = item.shape[0]

        the_list = getattr(self, list_name)
        if name in the_list:
            logging.debug(f"overwriting per_frame attr {name}")
        else:
            the_list += [name]

        setattr(self, name, item)
        logging.debug(f"add a pointer of {name} to the {list_name}")

    def include_frames(self, accept_id=None):

        if accept_id is None:
            return

        for k in self.per_frame_attrs:
            new_mat = getattr(self, k)
            new_mat = getattr(self, k)[accept_id]
            setattr(self, k, new_mat)
        self.nframes = len(accept_id)

    def fix_to_per_frame(self, key):

        if key not in self.fixed_attrs:
            return

        mat = self.expand_fixed_attr(key)
        setattr(self, key, mat)
        self.fixed_attrs.remove(key)
        self.per_frame_attrs += [key]

    def expand_fixed_attr(self, key):

        if key not in self.fixed_attrs:
            return

        mat = getattr(self, key)
        if isinstance(mat, np.ndarray):
            mat = np.tile(
                np.expand_dims(mat, axis=0), (self.nframes) + (1) * len(mat.shape)
            )
        else:
            mat = [mat] * self.nframes
        return mat

    @classmethod
    def from_file(cls, filename, update_dict={}, mapping={}):
        obj = load_file(
            supported_formats={"npz": "npz", "pickle": "pickle"}, filename=filename
        )
        if isinstance(obj, cls):
            obj.sanity_check()
            return obj
        return cls.from_dict(dict(obj), update_dict=update_dict, mapping=mapping)

    def to_dict(self):
        return {k: getattr(self, k) for k in self.keys}

    @classmethod
    def from_dict(cls, input_dict, update_dict={}, mapping={}):
        """

        requirement

        positions: nframes x ?

        optional:
            cells
            forces

        species, or symbols

        """
        trj = cls()

        backward_remap = {
            POSITION: POSITION + "s",
            FORCE: FORCE + "S",
            CELL: CELL + "s",
            TOTAL_ENERGY: "energies",
        }

        input_dict = {k: v for k, v in input_dict.items()}
        for new_name, original_name in mapping.items():
            input_dict[new_name] = input_dict.pop(original_name)
        for new_name, original_name in backward_remap.items():
            if original_name in input_dict:
                input_dict[new_name] = input_dict.pop(original_name)
        input_dict.update(update_dict)

        trj.nframes = input_dict[POSITION].shape[0]

        if CELL in input_dict:
            input_dict[CELL] = convert_cell_format(trj.nframes, input_dict[CELL])
        for k in [POSITION, FORCE]:
            if k in input_dict:
                input_dict[k] = input_dict[k].reshape([trj.nframes, -1, 3])
        trj.natoms = input_dict[POSITION].shape[1]

        for k in cls.stat_keys:
            if k in input_dict:
                setattr(trj, k, input_dict[k])

        for k in input_dict:
            found = False
            for attr in ["per_frame", "metadata", "fixed"]:
                input_list = input_dict.get(f"{attr}_attrs", [])
                if k in input_list and not found:
                    trj.add_field(f"{attr}_attrs", k, input_dict[k])
                    found = True
            for attr in ["per_frame", "metadata", "fixed"]:
                default_list = getattr(cls, f"default_{attr}_keys", [])
                if k in default_list and not found:
                    trj.add_field(f"{attr}_attrs", k, input_dict[k])
                    found = True

        trj.nframes = trj.position.shape[0]
        trj.natoms = trj.position.shape[1]

        remain_keys = set(list(input_dict.keys())) - set(trj.keys)
        for k in remain_keys:
            logging.debug(f"undefined attributes {k}, set to metadata")
            try:
                dim0 = input_dict[k].shape[0]
            except:
                dim0 = -1
            if dim0 == trj.nframes:
                trj.add_field(PER_FRAME_ATTRS, k, input_dict[k])
            else:
                trj.add_field(METADATA_ATTRS, k, input_dict[k])

        trj.sanity_check()
        return trj

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
        save_file(
            self.to_dict(),
            supported_formats={"npz": "npz", "pickle": "pickle"},
            filename=name,
            enforced_format=format,
        )

    @classmethod
    def stack(cls, trjs, safe_mode=True, order=None):
        """
        add all frames from another trajectory instance
        """

        nframes = np.sum([trj.nframes for trj in trjs])
        if nframes <= 0:
            return Trajectory()
        if len(trjs) == 1:
            trj = Trajectory()
            trj.copy(trjs[0])
            return trj

        if order is not None:
            for i, trj in enumerate(trjs):
                trj.reorder_atoms(order[i])

        natoms = set([trj.natoms for trj in trjs])
        if len(natoms) != 1:
            raise ValueError(f"cannot merge trjs with different numbers {natoms}")

        if safe_mode:
            labels = []
            for trj in trjs:
                _order, new_label = species_to_order_label(trj.species)
                labels += [new_label]
            labels = set(labels)
            if len(labels) != 1:
                raise ValueError(f"cannot merge trjs with different species {labels}")

        trj0 = trjs[0]
        d = trj0.to_dict()
        for k in trj0.per_frame_attrs:
            items = [getattr(trj, k) for trj in trjs]
            try:
                mat = np.stack(items, axis=0)
            except Exception as e:
                raise RuntimeError("fail", k, set([item.shape for item in items]), e)
            d[k] = mat
        d["nframes"] = nframes
        d["natoms"] = list(natoms)[0]
        trj = Trajectory.from_dict(d)
        return trj

    def add_trj(self, trj, safe_mode=True, order=None):
        """
        add all frames from another trajectory instance
        """

        if trj.nframes <= 0:
            return

        if order is not None:
            trj.reorder_atoms(order)

        if self.nframes == 0:
            self.copy(trj)
        else:
            if self.natoms != trj.natoms:
                raise ValueError(
                    f"cannot merge two trj with different numbers {self.natoms}, {trj.natoms}"
                )
            if safe_mode:
                if not all(trj.species == self.species):
                    _order, new_label = species_to_order_label(trj.species)
                    _, old_label = species_to_order_label(self.species)
                    if new_label != old_label:
                        raise ValueError(f"cannot merge two trj with different numbers")
                    if order is None:
                        trj.reorder_atoms(_order)

            for k in self.per_frame_attrs:
                item = getattr(trj, k)
                ori_item = getattr(self, k)
                try:
                    mat = np.append(ori_item, item, axis=0)
                except Exception as e:
                    raise RuntimeError("fail", k, item.shape, ori_item.shape, e)
                setattr(self, k, mat)

            self.copy_metadata(trj, exception=["name", "nframes", "natom", "filenames"])

            self.nframes += trj.nframes

    def reshape(self, key, shape):

        mat = getattr(self, k)
        if isinstance(mat, np.ndarray):
            setattr(self, k, mat.reshape(tuple([self.nframes]) + shape))

    def flatten(self):

        for k in self.per_frame_attrs:
            self.reshape(k, tuple([-1]))
        for k in self.fixed_attrs:
            self.reshape(k, tuple([-1]))

    def extract_frames(self, frame_list):

        trj = type(self)()
        trj.copy(self)

        trj.include_frames(frame_list)
        trj.name = f"{self.name}_{trj.nframes}"

        return trj

    def reorder_atoms(self, order):

        if len(order) > self.natoms:
            logging.error(
                f"{len(order)} order should be smaller than {self.natoms} lines"
            )
            raise RuntimeError()

        for k in self.per_frame_attrs:
            ori_item = getattr(self, k)
            if len(ori_item.shape) > 1:
                if ori_item.shape[1] == self.natoms:
                    item = np.swapaxes(ori_item, 0, 1)
                    item = item[order]
                    item = np.swapaxes(item, 0, 1)
                    setattr(self, k, item)

        self.species = np.array(self.species)[order]

        natoms = self.position.shape[1]

        if natoms != self.natoms:
            logging.info(f"extract_frames {self.natoms} lines to {natoms} lines")
            self.natoms = natoms

        self.sanity_check()

    def shuffle(self):

        frame_list = np.random.permutation(self.nframes)

        for k in self.per_frame_attrs:
            setattr(self, k, getattr(self, k)[frame_list])

    def copy(self, trj):
        for k in trj.keys:
            setattr(self, k, deepcopy(getattr(trj, k)))
        self.sanity_check()
