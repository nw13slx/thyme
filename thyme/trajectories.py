"""
Data structure that contains a collection of trajectory objects

Lixin Sun (Harvard University)
2020
"""

from copy import deepcopy
import logging
import numpy as np

from .trajectory import Trajectory
from thyme.utils.atomic_symbols import species_to_order_label
from thyme.utils.savenload import save_file, load_file


def dummy_comp(trj1, trj2):
    return True


class Trajectories:
    def __init__(self):

        self.nframes = 0
        self.ntrjs = 0
        self.alltrjs = {}
        self._iter_index = 0
        self.per_frame_attrs = []
        self.trj_id = None
        self.in_trj_id = None
        self.global_id = None

    def __repr__(self) -> str:
        return f"Trajectories with {len(self.alltrjs)} trj"

    def __str__(self):

        s = f"{len(self.alltrjs)} trajectories with {len(self)} frames\n"
        for name in self.alltrjs:
            s += f"----{name}----\n"
            s += f"{self.alltrjs[name]}\n"
        return s

    def __len__(self):
        return self.nframes

    def construct_id_list(self, force_run=False):

        if self.trj_id is not None and not force_run:
            max_trj = np.max(self.trj_id)
            max_frame = np.max(self.global_id)
            if max_trj == (self.ntrjs - 1) and max_frame == (self.nframes - 1):
                return

        self.trj_id = np.zeros(self.nframes, dtype=int)
        self.in_trj_id = np.zeros(self.nframes, dtype=int)
        self.global_id = np.arange(self.nframes)
        count = 0
        for id_trj, trj in enumerate(self.alltrjs.values()):
            nframes = trj.nframes
            self.trj_id[count : count + nframes] += id_trj
            self.in_trj_id[count : count + nframes] += np.arange(nframes)
            count += nframes

    def __getitem__(self, key):
        return self.alltrjs[key]

    def __iter__(self):
        return self

    def __next__(self):

        self._iter_index = getattr(self, "_iter_index", 0)

        if self._iter_index >= len(self):
            raise StopIteration
        self._iter_index += 1
        return self.get_frame(self._iter_index - 1)

    def get_frame(self, idx, keys=None):

        n_attrs = len(self)
        if idx >= n_attrs:
            raise ValueError(f"frame index overflow {n_attrs}")
        trj_id = self.trj_id[idx]
        frame_id = self.in_trj_id[idx]
        trj = list(self.alltrjs.values())[trj_id]
        trj_name = list(self.alltrjs.keys())[trj_id]
        return dict(name=trj_name, **trj.get_frame(frame_id, keys=keys))

    def get_attrs(self, key):

        self.construct_id_list()
        for id_trj, trj in enumerate(self.alltrjs.values()):
            attr = getattr(trj, key, None)
            if attr is None:
                raise ValueError(f"not all trjs has attr {attr}")

        array = []
        for id_trj, trj in enumerate(self.alltrjs.values()):
            sub_array = trj.get_attr(key)
            array += [sub_array]
        if len(array[0].shape) <= 1:
            array = np.hstack(array)
        else:
            array = np.vstack(array)
        return array[self.global_id]

    def include_frames(self, accept_id=None):

        if accept_id is None:
            return
        self.construct_id_list()

        self.trj_id = self.trj_id[accept_id]
        self.in_trj_id = self.in_trj_id[accept_id]
        self.global_id = self.global_id[accept_id]

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

        return {name: trj.to_dict() for name, trj in self.alltrjs.items()}

    @classmethod
    def from_file(cls, name: str, format: str = None, preserve_order: bool = False):
        """
        pickle format: previous objects saved as pickle format
        """

        obj = load_file(
            supported_formats={"npz": "npz", "pickle": "pickle"},
            filename=name,
            enforced_format=format,
        )
        if isinstance(obj, Trajectories):
            return obj
        return cls.from_dict(dict(obj))

    @staticmethod
    def from_dict(dictionary: dict, merge=False, preserve_order=False):
        """
        convert dictionary to a Trajectory instance
        """

        trjs = Trajectories()

        for name, trj_dict in dictionary.items():
            if not isinstance(trj_dict, dict):
                trj_dict = trj_dict.item()
            trj = Trajectory.from_dict(trj_dict)
            trjs.add_trj(trj, name=name, merge=merge, preserve_order=preserve_order)

        return trjs

    def pop_trj(self, name):
        trj = self.alltrjs.pop(name, None)
        if trj is not None:
            self.nframes -= trj.nframes
            self.ntrjs -= 1

    def add_trj(
        self,
        trj,
        name=None,
        merge=False,
        preserve_order=False,
        metadata_compare=dummy_comp,
        save_mode=True,
    ):
        if len(self.alltrjs) == 0:
            self.per_frame_attrs = deepcopy(trj.per_frame_attrs)
        elif save_mode:
            nterms = len(self.per_frame_attrs)
            intersection = set(self.per_frame_attrs).intersection(trj.per_frame_attrs)
            if len(intersection) != nterms:
                print(self.per_frame_attrs)
                print(trj.per_frame_attrs)
                raise RuntimeError(f"not enough per_frame_attrs")

        if not merge:

            if name in self.alltrjs:
                name=f"{name}_{len(self.alltrjs)}"
            elif name is None and trj.name not in self.alltrjs:
                name = trj.name
            elif name is None:
                name = f"{len(self.alltrjs)}"

            self.alltrjs[name] = trj
            self.nframes += trj.nframes
            self.ntrjs += 1
            return

        # order trj by element
        order, label = species_to_order_label(trj.species)

        if name is None:
            stored_label = None
            for _label, oldtrj in self.alltrjs.items():
                if metadata_compare(trj, oldtrj):
                    stored_label = _label
                    break
            if stored_label is None:
                stored_label = label
        else:
            stored_label = name+"_"+label
            label = name+"_"+label


        if stored_label not in self.alltrjs:
            newtrj = Trajectory()
            newtrj.name = np.copy(stored_label)
            self.alltrjs[stored_label] = newtrj
        else:
            oldtrj = self.alltrjs[stored_label]
            if metadata_compare(trj, oldtrj):
                logging.debug(f"! Metadata is exactly the same. Merge to {stored_label}")
            else:
                stored_label, last_label = obtain_store_label(
                    last_label="NA0",
                    label=label,
                    alldata=self.alltrjs,
                    preserve_order=True,
                )
                self.alltrjs[stored_label] = Trajectory()
                logging.debug(f"! Metadata is not the same. Not merge. Buil {stored_label}")

        self.alltrjs[stored_label].add_trj(trj, save_mode=False, order=order)
        self.nframes += trj.nframes
        self.ntrjs += 1
        return stored_label

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
        for trj in trjs.alltrjs.values():
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
