import logging
import numpy as np

from fmeee.utils.atomic_symbols import species_to_idgroups

def sort_by_force(trj, pred_label, chosen_species: str=None):
    """
    chosen_specie: str, only sort the force error among the atoms with
                        the chosen_species symbol
    """

    forces = trj.forces
    pred = getattr(trj, pred_label)
    dfs = forces-pred

    if chosen_species is None:
        maxdfs = np.max(np.abs(dfs.reshape([trj.nframes, -1])), axis=1)
        ids = np.argsort(maxdf)
        return ids, maxdf
    else:
        species, idgroups = species_to_idgroups(trj.species)
        iele = species.index(chosen_species)

        df = dfs[:, idgroups[iele], :]
        maxdf = np.max(np.abs(df.reshape([trj.nframes, -1])), axis=1)
        ids = np.argsort(maxdf)
        return ids, maxdf

def sort_by_energy(trj, pred_label):
    """
    """

    energies = trj.energies
    pred = getattr(trj, pred_label)
    de = energies-pred

    maxdfs = np.max(np.abs(de))
    return np.argsort(maxdfs)
