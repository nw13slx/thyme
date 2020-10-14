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
                ncount=len([idx for idx in range(trj.natom) if trj.symbols[i][idx]==chosen_specie])
                if ncount >= chosen_count:
                    return i
        else:
            ncount=len([idx for idx in range(trj.natom) if trj.species[idx]==chosen_specie])
            if ncount <= chosen_count:
                return -1
            else:
                return sorted_id[0]
    else:
        return sorted_id[0]
