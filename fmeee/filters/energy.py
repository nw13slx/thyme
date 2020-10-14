import logging
import numpy as np

from fmeee.utils.atomic_symbols import species_to_idgroups

def lowe(trj, chosen_species=None, chosen_count=0):
    """
    """

    sorted_id = np.argsort(trj.energies)
    if chosen_species is not None:
        if 'natoms' in trj.per_frame_attrs:
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
