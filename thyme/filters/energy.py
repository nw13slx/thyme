import logging
import numpy as np

from thyme.utils.atomic_symbols import species_to_idgroups, species_to_dict
from thyme.trajectory import PaddedTrajectory


def rm_sudden_drop(trj, thredshold):
    """"""

    ref = trj.energies[0]
    upper_bound = trj.nframes
    for i in range(trj.nframes):
        if trj.energies[i] < (ref - thredshold):
            upper_bound = i
            break
    if upper_bound != trj.nframes:
        logging.info(
            f" later part of the trj from {upper_bound} will be drop "
            "due to a sudden potential energy drop"
        )
    return np.arange(upper_bound)


def sort_e(trj, chosen_specie=None, chosen_count=0):
    """"""

    sorted_id = np.argsort(trj.energies)
    if chosen_specie is not None:
        if isinstance(trj, PaddedTrajectory):
            for i in sorted_id:
                ncount = len(
                    [
                        idx
                        for idx in range(trj.natoms[i])
                        if trj.symbols[i][idx] == chosen_specie
                    ]
                )
                if ncount == chosen_count:
                    return i
        else:
            ncount = len(
                [idx for idx in range(trj.natom) if trj.species[idx] == chosen_specie]
            )
            if ncount <= chosen_count:
                return -1
            else:
                return sorted_id
    else:
        return sorted_id


def lowe(trj, chosen_specie=None, chosen_count=0):
    """"""

    sorted_id = np.argsort(trj.energies)
    if chosen_specie is not None:
        if isinstance(trj, PaddedTrajectory):
            for i in sorted_id:
                ncount = len(
                    [
                        idx
                        for idx in range(trj.natoms[i])
                        if trj.symbols[i][idx] == chosen_specie
                    ]
                )
                if ncount == chosen_count:
                    return i
        else:
            ncount = len(
                [idx for idx in range(trj.natom) if trj.species[idx] == chosen_specie]
            )
            if ncount <= chosen_count:
                return -1
            else:
                return sorted_id[0]
    else:
        return sorted_id[0]


def rm_duplicate(trj):
    """
    remove top 3 energy, and then remove duplicated
    """

    sorted_id = np.argsort(trj.energies)[:-3]
    keep_id = []
    last_id = sorted_id[0]
    for i, idx in enumerate(sorted_id[1:]):
        if trj.energies[idx] != trj.energies[last_id]:
            keep_id += [idx]
            last_id = idx
        else:
            logging.info(f"remove duplicate energy {trj.energies[last_id]}")

    return keep_id

def fit_energy_shift(trjs):

    print("hello")
    sorted_trjs = trjs.remerge()

    x = []
    y = []
    species = set()
    for trj in sorted_trjs:
        symbol_dict = species_to_dict(sorted_trj.species)
        x += [symbol_dict]
        species = species.union(set(list(symbol_dict.keys())))
        y += [np.min(trj.energies)]

    allx = []
    for _x, _y in zip(x, y):

        order_x = [ _x.get(ele, 0)  for ele in species]
        allx += [order_x]

    return np.vstack(allx), np.array(y).reshape([-1, 1]), sorted_trjs
