import logging
import numpy as np

from thyme.utils.atomic_symbols import species_to_idgroups, species_to_dict
from thyme.trajectory import PaddedTrajectory


def even_hist(trj, max_count, bin_width, max_apperance=2, max_e=None):

    if len(trj) < max_count:
        logging.info(f"trajectory frames {trj.nframes} < {max_count}")
        return np.arange(trj.nframes)

    e = trj.energies
    if max_e is None:
        max_e = np.max(e)

    bins = np.arange(np.min(e), max_e + bin_width, bin_width)

    hist, bins = np.histogram(trj.energies, bins=bins)
    n_bins = len(np.where(hist > 0)[0])
    c_each_bin = max_count // n_bins
    keep_ids = []
    skip_idx = []
    for idx, h in enumerate(hist):
        if h == 0:
            skip_idx += [idx]
            continue

        left = bins[idx]
        right = bins[idx + 1]
        id1 = set(np.where(trj.energies > left)[0])
        id2 = set(np.where(trj.energies < right)[0])
        ids = np.array(list(id1.intersection(id2)))
        n = len(ids)
        if n < c_each_bin / max_apperance:
            draw = np.arange(n * max_apperance) // max_apperance
            keep_ids += [ids[draw]]
            skip_idx += [idx]
            print(left, right, n, len(draw))

    c_each_bin = (max_count - len(np.hstack(keep_ids))) // (len(hist)-len(skip_idx))
    for idx, h in enumerate(hist):
        if idx in skip_idx:
            continue

        left = bins[idx]
        right = bins[idx + 1]
        id1 = set(np.where(trj.energies > left)[0])
        id2 = set(np.where(trj.energies < right)[0])
        ids = np.array(list(id1.intersection(id2)))
        n = len(ids)
        draw = np.random.randint(n, size=c_each_bin)
        keep_ids += [ids[draw]]

        print(left, right, n, len(draw))

    keep_ids = np.hstack(keep_ids)
    return keep_ids


def remove_max_force(trj, max_force=20):
    """
    remove frames with forces larger than max_force
    """
    return [i for i in range(trj.nframes) if np.max(np.abs(trj.forces[i])) < max_force]


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


def fit_energy_shift(trjs, mode="min"):

    x = []
    y = []
    species = set()
    for trj in trjs:
        symbol_dict = species_to_dict(trj.species)
        species = species.union(set(list(symbol_dict.keys())))
        if mode == "min":
            x += [symbol_dict]
            y += [np.min(trj.energies)]
        elif mode == "max":
            x += [symbol_dict]
            y += [np.max(trj.energies)]
        elif mode.endswith("%"):

            percentage = float(mode[:-1])

            sort_e = np.argsort(trj.energies)

            up_idx = int(np.ceil(len(sort_e) * percentage / 100.0))

            for idx in sort_e[:up_idx]:
                x += [symbol_dict]
                y += [trj.energies[sort_e[idx]]]

    allx = []
    for _x, _y in zip(x, y):

        order_x = [_x.get(ele, 0) for ele in species]
        allx += [order_x]

    return np.vstack(allx), np.array(y).reshape([-1, 1])
