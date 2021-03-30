"""
Data structure that contains a collection of trajectory objects

Lixin Sun (Harvard University)
2020
"""

import logging
import numpy as np
import pickle

from collections import Counter

from thyme.trajectory import Trajectory, PaddedTrajectory
from thyme.utils.atomic_symbols import species_to_order_label
from thyme.utils.save import sort_format


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

        supported_formats = [
            "pickle",
            "padded_mat.npz",
            "npz",
            "padded.xyz",
            "xyz",
            "poscar",
        ]
        format, name = sort_format(supported_formats, format, name)

        if format == "pickle":
            with open(name, "wb") as f:
                pickle.dump(self, f)
        elif format == "padded_mat.npz":
            self.save_padded_matrices(name)
        elif format == "npz":
            self.save_npz(name)
        elif format == "padded.xyz":
            trj = self.to_padded_trajectory()
            trj.save(name, format)
        elif format == "xyz":
            for trj in self.alldata.values():
                trj.save(f"{trj.name}_{name}", format)
        elif format == "poscar":
            for trj in self.alldata.values():
                trj.save(f"{trj.name}_{name}", format)
        else:
            raise NotImplementedError(
                f"Output format {format} not supported:"
                f" try from {supported_formats}"
            )
        logging.info(f"save as {name}")

    def to_dict(self):

        alldata = {}

        for name, trj in self.alldata.items():
            alldata[name] = trj.to_dict()

        return alldata

    def to_padded_trajectory(self):

        init_trj = PaddedTrajectory()
        for trj in self.alldata.values():
            init_trj.add_trj(trj)
        return init_trj

    def add_trj(self, trj, name=None):

        if isinstance(trj, Trajectories):
            self.alldata.update(trj.alldata)
        else:
            if name in self.alldata:
                logging.info(f"warning, overwriting trj with name {name}")

            if name is None:
                name = trj.name
            self.alldata[name] = trj

    def save_padded_matrices(self, name: str):

        if ".npz" != name[-4:]:
            name += ".npz"

        init_trj = self.to_padded_trajectory()
        init_trj.save(name)

    def save_npz(self, name: str):

        if ".npz" != name[-4:]:
            name += ".npz"

        dictionary = self.to_dict()
        np.savez(name, **dictionary)

    @staticmethod
    def from_file(name: str, format: str = None, preserve_order: bool = False):
        """
        pickle format: previous objects saved as pickle format
        padded_mat.npz: contains matrices that can be parsed by PaddedTrajectory
                        from file loader. and then the frames are partitioned
                        such that eacy trajectory has the same number of atoms
                        and same order of species
        """

        supported_formats = ["pickle", "padded_mat.npz"]  # npz

        format, newname = sort_format(supported_formats, format, name)

        if format == "pickle":
            with open(name, "rb") as f:
                trjs = pickle.load(f)
            return trjs
        elif format == "padded_mat.npz":
            dictionary = dict(np.load(name, allow_pickle=True))
            return Trajectories.from_padded_matrices(
                dictionary, preserve_order=preserve_order
            )
        else:
            raise NotImplementedError(
                f"Output format not supported:" f" try from {supported_formats}"
            )

    @staticmethod
    def from_dict(dictionary: dict, merge=True):
        """
        convert dictionary to a Trajectory instance
        """

        raise NotImplementedError("this part need to be double check!")

        trjs = Trajectories()
        alldata = trjs.alldata

        trjnames = sorted(list(dictionary.keys()))

        for trjname in trjnames:
            try:
                data = dictionary[trjname].item()
                order, label = species_to_order_label(data["species"])
            except:
                data = dictionary[trjname].item()
                order, label = species_to_order_label(data["species"])

            logging.info(f"read {trjname} from dict formula {label}")

            if merge:
                if label not in alldata:
                    alldata[label] = Trajectory()
                    alldata[label].python_list = True
                alldata[label].add_trj(Trajectory.from_dict(data))
            else:
                alldata[trjname] = Trajectory.from_dict(data)

        for label in alldata:
            alldata[label].convert_to_np()
            alldata[label].name = f"{label}"
            logging.info(f"from dict {repr(alldata[label])}")

        return trjs

    @property
    def nframes(self):
        nframes = 0
        for trj in self.alldata.values():
            nframes += trj.nframes
        return nframes

    def remerge(self, preserve_order=False, metadata_compare=dummy_comp):

        trjs = Trajectories()

        # trj.nframes = ptrj.nframes
        # trj.symbols = ptrj.symbols

        last_label = None
        label = None
        curr_label_count = 0
        last_label_count = 0
        alldata = trjs.alldata
        for trj in self.alldata.values():

            # obtain label
            order, label = species_to_order_label(trj.species)
            natom = trj.natom
            newtrj = Trajectory()
            newtrj.copy(trj)
            newtrj.reorder(order)

            stored_label, last_label = obtain_store_label(
                last_label, label, alldata, preserve_order
            )

            if stored_label not in alldata:
                newtrj.name = np.copy(stored_label)
                alldata[stored_label] = newtrj
            else:
                if metadata_compare(trj, newtrj):
                    logging.info("! True merge")
                    alldata[stored_label].add_trj(newtrj)
                else:
                    logging.info("! False merge")
                    newtrj.name = stored_label
                    stored_label, last_label = obtain_store_label(
                        "NA0", label, alldata, True
                    )
                    alldata[stored_label] = newtrj

        for i in alldata:
            trj = alldata[i]
            alldata[i].name = i
            logging.info(f"found one type of formula {trj.name} with key {i}")
            logging.info(f"add {repr(trj)}")

        return trjs

    @staticmethod
    def from_padded_trajectory(ptrj: dict, preserve_order=False):

        trjs = Trajectories()

        nframes = ptrj.nframes
        symbols = ptrj.symbols

        last_label = None
        label = None
        curr_label_count = 0
        last_label_count = 0
        alldata = {}
        for iconfig in range(nframes):

            # obtain label
            order, label = species_to_order_label(symbols[iconfig])
            natom = ptrj.natoms[iconfig]

            stored_label, last_label = obtain_store_label(
                last_label, label, alldata, preserve_order
            )

            if stored_label not in alldata:
                alldata[stored_label] = [label, [iconfig], [order]]
            else:
                alldata[stored_label][1].append(iconfig)
                alldata[stored_label][2].append(order)

        for stored_label in alldata:

            label = alldata[stored_label][0]
            configs = alldata[stored_label][1]
            orders = alldata[stored_label][2]

            trj = ptrj.skim(configs)
            trj.reorder(orders)

            newtrj = trj.to_Trajectory()
            newtrj.name = label

            trjs.add_trj(newtrj, stored_label)
            logging.info(f"{label} {configs}")
            logging.info(f"found one type of formula {label}")
            logging.info(f"add {repr(newtrj)}")

        return trjs

    @staticmethod
    def from_padded_matrices(
        dictionary: dict, per_frame_attrs: list = None, preserve_order=False
    ):
        """
        Keys needed:

        positions  (n, m, 3)
        symbols    (n, m)
        natoms     (n, m)

        if preserve_order is off (default)
            all the configures that has the same number of species

        """

        ptrj = PaddedTrajectory.from_dict(dictionary, per_frame_attrs)

        return Trajectories.from_padded_trajectory(ptrj, preserve_order)


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
