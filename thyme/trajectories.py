"""
Data structure that contains a collection of trajectory objects

Lixin Sun (Harvard University)
2020
"""

from copy import deepcopy
import logging
import numpy as np
import pickle

from collections import Counter

from .trajectory import Trajectory
from thyme.utils.atomic_symbols import species_to_order_label
from thyme.utils.savenload import save_file, load_file


def dummy_comp(trj1, trj2):
    return True


class Trajectories:
    def __init__(self):
        self.alltrjs = {}
        self._iter_index = 0
        self.per_frame_attrs = []

    def __repr__(self) -> str:
        return f"Trajectories with {len(self)} trj"

    def __str__(self) -> str:
        s = repr(self)
        for name, trj in self.alltrjs:
            s += "\nname: {repr(trj)}"
        return s

    @property
    def nframes(self):
        nframes = 0
        for trj in self.alltrjs.values():
            nframes += trj.nframes
        return nframes

    def __len__(self):
        return len(self.alltrjs)

    def __str__(self):

        s = f"{len(self.alltrjs)} trajectories\n"
        for name in self.alltrjs:
            s += f"----{name}----\n"
            s += f"{self.alltrjs[name]}\n"
        return s

    def __getitem__(self, key):
        return self.alltrjs[key]

    def __iter__(self):
        return self

    def __next__(self):

        self._iter_index = getattr(self, "_iter_index", 0)

        n_attrs = len(self.alltrjs)
        if self._iter_index >= n_attrs:
            raise StopIteration
        key = list(self.alltrjs.keys())[self._iter_index]
        self._iter_index += 1
        return self.alltrjs[key]

    def save(self, name: str, format: str = None):

        if format in ["pickle", "npz"] or format is None:
            save_file(
                self.to_dict(),
                supported_formats={"npz": "npz", "pickle": "pickle"},
                filename=name,
                enforced_format=format,
            )
        elif format == "xyz":
            for trj in self.alltrjs.values():
                trj.save(f"{trj.name}_{name}", format)
        elif format == "poscar":
            for trj in self.alltrjs.values():
                trj.save(f"{trj.name}_{name}", format)
        else:
            raise NotImplementedError(
                f"Output format {format} not supported:"
                f" try from pickle, xyz, poscar"
            )
        logging.info(f"save as {name}")

    def to_dict(self):

        dictionary = {}

        for name, trj in self.alltrjs.items():
            dictionary[name] = trj.to_dict()

        return dictionary

    @staticmethod
    def from_file(name: str, format: str = None, preserve_order: bool = False):
        """
        pickle format: previous objects saved as pickle format
        """

        obj = load_file(
            supported_formats={"npz": "npz", "pickle": "pickle"},
            filename=name,
            enforced_format="npz",
        )
        if isinstance(obj, dict):
            return Trajectories.from_dict(obj)
        return obj

    @staticmethod
    def from_dict(dictionary: dict, merge=True, preserve_order=False):
        """
        convert dictionary to a Trajectory instance
        """

        trjs = Trajectories()

        for name, trj_dict in dictionary.items():
            trj = Trajectory.from_dict(trj_dict)
            trjs.add_trj(trj, merge=merge, preserve_order=preserve_order)

        return trjs

    @property
    def nframes(self):
        nframes = 0
        for trj in self.alltrjs.values():
            nframes += trj.nframes
        return nframes

    def add_trj(
        self,
        trj,
        name=None,
        merge=False,
        preserve_order=False,
        metadata_compare=dummy_comp,
        save_mode=True,
    ):
        if len(self) == 0:
            self.per_frame_attrs = deepcopy(trj.per_frame_attrs)
        elif save_mode:
            nterms = len(self.per_frame_attrs)
            intersection = set(self.per_frame_attrs).intersection(trj.per_frame_attrs)
            if len(intersection) != nterms:
                raise RuntimeError(f"not enough per_frame_attrs")

        if not merge:
            if name in self.alltrjs:
                logging.info(f"warning, overwriting trj with name {name}")

            if name is None and trj.name not in self.alltrjs:
                name = trj.name
            elif name is None:
                name = len(self)
            self.alltrjs[name] = trj
            return

        # order trj by element
        order, label = species_to_order_label(trj.species)

        stored_label, last_label = obtain_store_label(
            last_label=None,
            label=label,
            alldata=self.alltrjs,
            preserve_order=preserve_order,
        )

        if stored_label not in self.alltrjs:
            newtrj = Trajectory()
            newtrj.name = np.copy(stored_label)
            self.alltrjs[stored_label] = newtrj
        else:
            oldtrj = self.alltrjs[stored_label]
            if metadata_compare(trj, oldtrj):
                logging.debug("! Metadata is exactly the same. Merge")
            else:
                logging.debug("! Metadata is not the same. Not merge")
                stored_label, last_label = obtain_store_label(
                    last_label="NA0",
                    label=label,
                    alldata=self.alltrjs,
                    preserve_order=True,
                )
                self.alltrjs[stored_label] = Trajectory()

        self.alltrjs[stored_label].add_trj(trj, save_mode=False, order=order)

    def add_trjs(
        self,
        trjs,
        merge=False,
        preserve_order=False,
        metadata_compare=dummy_comp,
    ):
        nterms = len(self.per_frame_attrs)
        intersection = set(self.per_frame_attrs).intersection(trjs.per_frame_attrs)
        if len(intersection) != nterms:
            raise RuntimeError(f"not enough per_frame_attrs")
        for trj in trjs:
            self.add_trj(
                trj,
                name=None,
                merge=merge,
                preserve_order=preserve_order,
                metadata_compare=metadata_compare,
                save_mode=False,
            )

    def merge(self, preserve_order=False, metadata_compare=dummy_comp):

        trjs = self.remerge()
        del self.alltrjs
        self.alltrjs = trjs.alltrjs

    def remerge(self, preserve_order=False, metadata_compare=dummy_comp):

        trjs = Trajectories()

        for trj in self.alltrjs.values():
            trjs.add_trj(
                trj,
                name=None,
                merge=True,
                preserve_order=preserve_order,
                metadata_compare=metadata_compare,
            )

        return trjs


def obtain_store_label(last_label, label, alldata, preserve_order):
    stored_label = label
    if preserve_order:

        count = -1
        # find all the previous trajectories
        for l in alldata:
            if "_" in l:
                line_split = l.split("_")
            else:
                line_split = l
            if label == line_split[0]:
                _count = int(line_split[1])
                if _count > count:
                    count = _count
        if label != last_label:
            count += 1
            last_label = label

        stored_label = f"{label}_{count}"
    return stored_label, last_label
