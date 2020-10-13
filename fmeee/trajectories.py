import numpy as np

from collections import Counter

from fmeee.trajectory import Trajectory

class Trajectories():

    def __init__(self):
        self.alldata = {}

    @staticmethod
    def from_dict(dictionary:dict):
        """
        convert dictionary to a Trajectory instance
        """
        pass
        # alldata = dict(np.load("alldata.npz", allow_pickle=True))

        # trjnames = sort_filenames(alldata)

        # positions = []
        # forces = []
        # energies = []
        # cells = []
        # ntrjs = 0
        # merge_data = {}
        # for trjname in trjnames:
        #     data = alldata[trjname].item()
        #     count = dict(Counter(data['species']))
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

        for i in range(nframes):

            # obtain label
            order, label = species_to_order_label(symbols[i])
            natom = dictionary['natoms'][i]

            if label not in alldata:
                alldata[label] = Trajectory()
                alldata[label].python_list = True

            alldata[label].add_frame_from_dict(dictionary, nframes,
                                               i=i, attributes=per_frame_attr,
                                               idorder=order)

        for label in alldata:
            alldata[label].convert_to_np()

        return trjs


def species_to_order_label(symbol):

    count = dict(Counter(symbol))
    if 'NA' in count:
        del count['NA']
    order = []
    symbol_list = np.sort(list(count.keys()))
    for k in symbol_list:
        order += [[i for i, s in enumerate(symbol) if s==k]]
    order = np.hstack(order)
    label = "".join([f"{k}{count[k]}" for k in symbol_list])
    return order, label
