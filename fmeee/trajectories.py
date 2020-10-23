"""
Data structure that contains a collection of trajectory objects

Lixin Sun (Harvard University)
2020
"""

import logging
import numpy as np
import pickle

from collections import Counter

from fmeee.trajectory import Trajectory, PaddedTrajectory
from fmeee.utils.atomic_symbols import species_to_order_label
from fmeee.utils.save import sort_format

class Trajectories():

    def __init__(self):
        self.alldata = {}

    def save(self, name: str, format: str = None):

        supported_formats = ['pickle', 'padded_mat.npz', 'npz', 'padded.xyz',
                             'xyz', 'poscar']
        format, name = sort_format(supported_formats, format, name)

        if format == 'pickle':
            with open(name, 'wb') as f:
                pickle.dump(self, f)
        elif format == 'padded_mat.npz':
            self.save_padded_matrices(name)
        elif format == 'npz':
            self.save_npz(name)
        elif format == 'padded.xyz':
            trj = self.to_padded_trajectory()
            trj.save(name, format)
        elif format == 'xyz':
            for trj in self.alldata.values():
                trj.save(f"{trj.name}_{name}", format)
        elif format == 'poscar':
            for trj in self.alldata.values():
                trj.save(f"{trj.name}_{name}", format)
        else:
            raise NotImplementedError(f"Output format {format} not supported:"
                                      f" try from {supported_formats}")

    def to_dict(self):

        alldata = {}

        for name, trj in self.alldata.items():
            alldata[name] = trj.to_dict()

        return alldata

    def to_padded_trajectory(self):

        max_atom = 0
        for trj in self.alldata.values():
            if trj.natom > max_atom:
                max_atom = trj.natom

        init_trj = PaddedTrajectory()
        for trj in self.alldata.values():
            ptrj = PaddedTrajectory.from_trajectory(trj, max_atom)
            logging.debug(f"padd {trj.name} to {ptrj}")
            init_trj.add_trj(ptrj)
        return init_trj


    def save_padded_matrices(self, name:str):

        if ".npz" != name[-4:]:
            name += '.npz'

        init_trj = self.to_padded_trajectory()
        init_trj.save(name)

    def save_npz(self, name:str):

        if ".npz" != name[-4:]:
            name += '.npz'

        dictionary = self.to_dict()
        np.savez(name, **dictionary)

    @staticmethod
    def from_file(name: str, format: str = None):
        """
        pickle format: previous objects saved as pickle format
        padded_mat.npz: contains matrices that can be parsed by PaddedTrajectory
                        from file loader. and then the frames are partitioned
                        such that eacy trajectory has the same number of atoms
                        and same order of species
        """

        supported_formats = ['pickle', 'padded_mat.npz'] # npz

        format, newname = sort_format(supported_formats, format, name)

        if format == 'pickle':
            with open(name, 'rb') as f:
                trjs = pickle.load(f)
            return trjs
        elif format == 'padded_mat.npz':
            dictionary = dict(np.load(name,
                                      allow_pickle=True))
            return Trajectories.from_padded_matrices(dictionary)
        else:
            raise NotImplementedError(f"Output format not supported:"
                                      f" try from {supported_formats}")


    @staticmethod
    def from_dict(dictionary:dict, merge=True):
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
                 order, label = species_to_order_label(data['species'])
            except:
                 data = dictionary[trjname].item()
                 order, label = species_to_order_label(data['species'])

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

        #     label = "".join([f"{k}{count[k]}" for k in np.sort(list(count.keys()))])
        #     sort_id = np.argsort(data['species'])
        #     if label not in merge_data:
        #         merge_data[label] = {}
        #         for k in ['positions', 'forces', 'energies', 'cells', 'history']:
        #             merge_data[label][k] = []
        #         merge_data[label]['species'] = np.sort(data['species'])

        #     nframes = data['positions'].shape[0]
        #     if nframes > 0:
        #         for k in ['positions', 'forces']:
        #             merge_data[label][k] += [(data[k].reshape([nframes, -1, 3])[:, sort_id, :]).reshape([nframes, -1])]
        #         for k in ['energies', 'cells']:
        #             merge_data[label][k] += [data[k]]
        #         names = [f"{trjname}_{i}" for i in range(nframes)]
        #         merge_data[label]['history'] += names
        #     ntrjs += 1

        # for label in merge_data:
        #     for k in ['positions', 'forces', 'cells']:
        #         merge_data[label][k] = np.vstack(merge_data[label][k])
        #     for k in ['energies']:
        #         merge_data[label][k] = np.hstack(merge_data[label][k])
        #     np.savez(f"all_{label}.npz", species=merge_data[label]['species'],
        #              positions=merge_data[label]['positions'],
        #              forces=merge_data[label]['forces'],
        #              energies=merge_data[label]['energies'],
        #              cells=merge_data[label]['cells'],
        #              names=merge_data[label]['history'])
        #     print(label, len(merge_data[label]['energies']), len(merge_data[label]['history']))

        # return 0

    @property
    def nframes(self):
        nframes = 0
        for trj in self.alldata.values():
            nframes += trj.nframes
        return nframes


    @staticmethod
    def from_padded_matrices(dictionary:dict,
                             per_frame_attr:list =None):
        """
        Keys needed:

        positions  (n, m, 3)
        symbols    (n, m)
        natoms     (n, m)

        """

        trjs = Trajectories()
        alldata = trjs.alldata


        nframes = dictionary['positions'].shape[0]
        max_atoms = dictionary['symbols'].shape[1]
        symbols = dictionary['symbols']

        if per_frame_attr is None:
            per_frame_attr = []
            for k in dictionary:
                try:
                    if dictionary[k].shape[0] == nframes:
                        per_frame_attr += [k]
                except Exception as e:
                    logging.debug(f"skip {k} because of {e}")

        for i in range(nframes):

            # obtain label
            order, label = species_to_order_label(symbols[i])
            natom = dictionary['natoms'][i]

            if label not in alldata:
                alldata[label] = Trajectory()
                alldata[label].name = label
                alldata[label].python_list = True

            alldata[label].add_frame_from_dict(dictionary, nframes,
                                               i=i, attributes=per_frame_attr,
                                               idorder=order)

        for label in alldata:
            alldata[label].convert_to_np()

        return trjs

