"""
Data structure that contains a collection of trajectory objects

Lixin Sun (Harvard University)
2020
"""

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
        self.alldata = {}
        self._iter_index = 0

    @property
    def nframes(self):
        nframes = 0
        for trj in self.alldata.values():
            nframes += trj.nframes
        return nframes

    def __len__(self):
        return len(self.alldata)

    def __str__(self):

        s = f"{len(self.alldata)} trajectories\n"
        for name in self.alldata:
            s += f"----{name}----\n"
            s += f"{self.alldata[name]}\n"
        return s

    def __getitem__(self, key):
        return self.alldata[key]

    def __iter__(self):
        return self

    def __next__(self):

        self._iter_index = getattr(self, "_iter_index", 0)

        n_attrs = len(self.alldata)
        if self._iter_index >= n_attrs:
            raise StopIteration
        key = list(self.alldata.keys())[self._iter_index]
        self._iter_index += 1
        return self.alldata[key]

    def save(self, name: str, format: str = None):

        if format in ["pickle", "npz"] or format is None:
            save_file(
                self.to_dict(),
                supported_formats={"npz": "npz", "pickle": "pickle"},
                filename=name,
                enforced_format=format,
            )
        elif format == "xyz":
            for trj in self.alldata.values():
                trj.save(f"{trj.name}_{name}", format)
        elif format == "poscar":
            for trj in self.alldata.values():
                trj.save(f"{trj.name}_{name}", format)
        else:
            raise NotImplementedError(
                f"Output format {format} not supported:"
                f" try from pickle, xyz, poscar"
            )
        logging.info(f"save as {name}")

    def to_dict(self):

        dictionary = {}

        for name, trj in self.alldata.items():
            dictionary[name] = trj.to_dict()

        return dictionary

    def to_trajectory(self):

        init_trj = Trajectory()
        for trj in self.alldata.values():
            init_trj.add_trj(trj)
        return init_trj

    def from_dict(self):
        pass

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
        for trj in self.alldata.values():
            nframes += trj.nframes
        return nframes

    def add_trj(
        self,
        trj,
        name=None,
        merge=False,
        preserve_order=False,
        metadata_compare=dummy_comp,
    ):

        if not merge:
            if isinstance(trj, Trajectories):
                self.alldata.update(trj.alldata)
            else:
                if name in self.alldata:
                    logging.info(f"warning, overwriting trj with name {name}")

                if name is None and trj.name not in self.alldata[name]:
                    name = trj.name
                elif name is None:
                    name = len(self)
                self.alldata[name] = trj
            return

        # order trj by element
        order, label = species_to_order_label(trj.species)

        stored_label, last_label = obtain_store_label(
            last_label=None,
            label=label,
            alldata=self.alldata,
            preserve_order=preserve_order,
        )

        if stored_label not in self.alldata:
            newtrj = Trajectory()
            newtrj.name = np.copy(stored_label)
            self.alldata[stored_label] = newtrj
        else:
            oldtrj = self.alldata[stored_label]
            if metadata_compare(trj, oldtrj):
                logging.debug("! Metadata is exactly the same. Merge")
            else:
                logging.debug("! Metadata is not the same. Not merge")
                stored_label, last_label = obtain_store_label(
                    last_label="NA0", label=label, alldata=self.alldata, preserve_order=True
                )
                self.alldata[stored_label] = Trajectory()

        self.alldata[stored_label].add_trj(trj, save_mode=False, order=order)

    def add_trjs(
        self,
        trjs,
        merge=False,
        preserve_order=False,
        metadata_compare=dummy_comp,
    ):
        for trj in trjs:
            self.add_trj(trj, name=None, merge=merge, preserve_order=preserve_order, metadata_compare=metadata_compare)
    
    def merge(self, preserve_order=False, metadata_compare=dummy_comp):

        trjs = self.remerge()
        del self.alldata
        self.alldata = trjs.alldata

    def remerge(self, preserve_order=False, metadata_compare=dummy_comp):

        trjs = Trajectories()

        for trj in self.alldata.values():
            trjs.add_trj(
                trj,
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
