import logging
import numpy as np

from thyme.utils.atomic_symbols import species_to_idgroups, species_to_dict
from thyme import Trajectory, Trajectories


def skew_hist(obj, max_count, bin_width, max_apperance=2, skew_factor=6, max_e=None):

    if len(obj) < max_count:
        logging.info(f"trajectory frames {obj.nframes} < {max_count}")
        return np.arange(obj.nframes)

    e = obj.total_energy
    if max_e is None:
        max_e = np.max(e)

    bins = np.arange(np.min(e), max_e + bin_width, bin_width)

    hist, bins = np.histogram(obj.total_energy, bins=bins)
    ids = np.where(hist > 0)[0]
    hist = hist[ids]
    bin_left = bins[ids]
    bin_right = bins[ids + 1]
    n_bins = len(hist)
    c_each_bin = np.ceil(
        max_count
        / (n_bins * (skew_factor + 2) / 2.0)
        * (-np.arange(n_bins) / n_bins * skew_factor + skew_factor + 1)
    )
    c_each_bin = np.array(c_each_bin, int)
    keep_ids = []
    skip_idx = []
    for idx, h in enumerate(hist):

        left = bin_left[idx]
        right = bin_right[idx]
        id1 = set(np.where(obj.total_energy > left)[0])
        id2 = set(np.where(obj.total_energy < right)[0])
        ids = np.array(list(id1.intersection(id2)))
        n = len(ids)
        if n < c_each_bin[idx] / max_apperance:
            draw = np.arange(n * max_apperance) // max_apperance
            keep_ids += [ids[draw]]
            skip_idx += [idx]
            print(left, right, n, len(draw))
    print("skip_idx", skip_idx)

    not_skip = list(set(np.arange(n_bins)) - set(skip_idx))

    hist = hist[not_skip]
    bin_left = bin_left[not_skip]
    bin_right = bin_right[not_skip]
    n_bins = len(hist)

    max_count -= len(np.hstack(keep_ids))
    c_each_bin = np.ceil(
        max_count
        / (n_bins * (skew_factor + 2) / 2.0)
        * (-np.arange(n_bins) / n_bins * skew_factor + skew_factor + 1)
    )
    c_each_bin = np.array(c_each_bin, int)
    for idx, h in enumerate(hist):

        left = bin_left[idx]
        right = bin_right[idx]
        id1 = set(np.where(obj.total_energy > left)[0])
        id2 = set(np.where(obj.total_energy < right)[0])
        ids = np.array(list(id1.intersection(id2)))
        n = len(ids)
        draw = np.random.randint(n, size=c_each_bin[idx])
        keep_ids += [ids[draw]]
        logging.debug(f"draw ({left}, {right}): {n}, {len(draw)}")

    keep_ids = np.hstack(keep_ids)
    return keep_ids


def even_hist(obj, max_count, bin_width, max_apperance=2, max_e=None):

    if len(obj) < max_count:
        logging.info(f"trajectory frames {obj.nframes} < {max_count}")
        return np.arange(obj.nframes)

    e = obj.total_energy
    if max_e is None:
        max_e = np.max(e)

    bins = np.arange(np.min(e), max_e + bin_width, bin_width)

    hist, bins = np.histogram(obj.total_energy, bins=bins)
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
        id1 = set(np.where(obj.total_energy > left)[0])
        id2 = set(np.where(obj.total_energy < right)[0])
        ids = np.array(list(id1.intersection(id2)))
        n = len(ids)
        if n < c_each_bin / max_apperance:
            draw = np.arange(n * max_apperance) // max_apperance
            keep_ids += [ids[draw]]
            skip_idx += [idx]
            print(left, right, n, len(draw))

    if len(keep_ids) == 0:
        return []

    c_each_bin = (max_count - len(np.hstack(keep_ids))) // (len(hist) - len(skip_idx))
    for idx, h in enumerate(hist):
        if idx in skip_idx:
            continue

        left = bins[idx]
        right = bins[idx + 1]
        id1 = set(np.where(obj.total_energy > left)[0])
        id2 = set(np.where(obj.total_energy < right)[0])
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

    ref = trj.total_energy[0]
    upper_bound = trj.nframes
    for i in range(trj.nframes):
        if trj.total_energy[i] < (ref - thredshold):
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

    sorted_id = np.argsort(trj.total_energy)
    if chosen_specie is not None:
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

    sorted_id = np.argsort(trj.total_energy)
    if chosen_specie is not None:
        ncount = len(
            [idx for idx in range(trj.natom) if trj.species[idx] == chosen_specie]
        )
        if ncount <= chosen_count:
            return -1
        else:
            return sorted_id[0]
    else:
        return sorted_id[0]


def thresholding(trj, cap=40):
    """
    remove top 3 energy, and then remove duplicated
    """

    min_e = np.min(trj.total_energy)

    return np.where(trj.total_energy < (min_e + cap))[0]


def rm_duplicate(trj):
    """
    remove top 3 energy, and then remove duplicated
    """

    sorted_e = np.argsort(trj.total_energy)[:-3]
    keep_id = []
    last_id = sorted_e[0]
    for i, idx in enumerate(sorted_e[1:]):
        if trj.total_energy[idx] != trj.total_energy[last_id]:
            keep_id += [idx]
            last_id = idx
        else:
            logging.info(f"remove duplicate energy {trj.total_energy[last_id]}")

    # preserve the original order
    return np.sort(keep_id)


def fit_energy_shift(trjs, mode="min"):

    x = []
    y = []
    species = set()
    for name, trj in trjs.alltrjs.items():
        symbol_dict = species_to_dict(trj.species)
        species = species.union(set(list(symbol_dict.keys())))
        if mode == "min":
            x += [symbol_dict]
            y += [np.min(trj.total_energy)]
        elif mode == "max":
            x += [symbol_dict]
            y += [np.max(trj.total_energy)]
        elif mode.endswith("%"):

            percentage = float(mode[:-1])

            sort_e = np.argsort(trj.total_energy)

            up_idx = int(np.ceil(len(sort_e) * percentage / 100.0))

            for idx in sort_e[:up_idx]:
                x += [symbol_dict]
                y += [trj.total_energy[sort_e[idx]]]

    allx = []
    for _x, _y in zip(x, y):

        order_x = [_x.get(ele, 0) for ele in species]
        allx += [order_x]

    return np.vstack(allx), np.array(y).reshape([-1, 1])
