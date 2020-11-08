import logging
import numpy as np

from fmeee.utils.atomic_symbols import species_to_idgroups
from fmeee.trajectory import PaddedTrajectory

def lowe(trj, chosen_specie=None, chosen_count=0):
    """
    """

    sorted_id = np.argsort(trj.energies)
    if chosen_specie is not None:
        if isinstance(trj, PaddedTrajectory):
            for i in sorted_id:
                ncount=len([idx for idx in range(trj.natoms[i]) if trj.symbols[i][idx]==chosen_specie])
                if ncount == chosen_count:
                    return i
        else:
            ncount=len([idx for idx in range(trj.natom) if trj.species[idx]==chosen_specie])
            if ncount <= chosen_count:
                return -1
            else:
                return sorted_id[0]
    else:
        return sorted_id[0]

def rm_duplicate(trj):
    """
    remove top 2 energy, and then remove duplicated
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
