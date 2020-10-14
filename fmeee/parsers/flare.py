import logging
import numpy as np

from math import inf

from flare.struc import Structure
from fmeee.trajectories import Trajectory, PaddedTrajectory


def configuration(trj, i):
    if isinstance(trj, PaddedTrajectory):
        natom = trj.natoms[i]
        structure = Structure(trj.cells[i].reshape([3, 3]),
                              trj.symbols[i][:natom],
                              trj.positions[i][:natom].reshape([natom, 3]),
                              forces=trj.forces[i][:natom].reshape([natom, 3]),
                              energy=trj.energies[i])
    else:
        structure = Structure(trj.cells[i].reshape([3, 3]), trj.species,
                              trj.positions[i].reshape([-1, 3]),
                              forces=trj.forces[i].reshape([-1, 3]),
                              energy=trj.energies[i])
    return structure


def add_to_model(filename, train_frame, gp_model, pre_train_env_per_species):

    trj = PaddedTrajectory.from_dict(
        dict(np.load(filename, allow_pickle=True)))

    structures = [ configuration(trj, i) for i in range(train_frame)]

    count = 0
    for structure in structures:

        for species_i in set(structure.coded_species):

            atoms_of_specie = list(set(structure.indices_of_specie(species_i)))

            n_at = len(atoms_of_specie)

            np.random.shuffle(atoms_of_specie)

            # Determine how many to add based on user defined cutoffs
            n_to_add = min(n_at, pre_train_env_per_species.get(
                species_i, inf), 2)
            if n_to_add > 0:
                gp_model.update_db(structure, structure.forces,
                                   custom_range=atoms_of_specie[:n_to_add])
                count += n_to_add

                logging.info(f"Add {n_to_add} Atoms with specie {species_i} to model"
                      f"; Total envs {count}")

    return [ configuration(trj, i) for i in range(train_frame, trj.nframes)]
