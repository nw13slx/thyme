import logging
import numpy as np
import pickle

from glob import glob
from os.path import getmtime

from flare.struc import Structure

from thyme.parsers.monty import read_pattern, read_table_pattern
from thyme.trajectory import PaddedTrajectory, Trajectory
from thyme.trajectories import Trajectories
from thyme.routines.folders import find_folders, find_folders_matching


def to_strucs(trj):
    structures = []
    if isinstance(trj, Trajectory) and not isinstance(trj, PaddedTrajectory):
        for i in range(trj.nframes):
            natom = trj.natoms[i]
            structure = Structure(
                cell=trj.cells[i].reshape([3, 3]),
                species=trj.species[:natom],
                positions=trj.positions[i][:natom].reshape([-1, 3]),
                forces=trj.forces[i][:natom].reshape([-1, 3]),
                energy=trj.energies[i],
            )
            structures += [structure]
    elif isinstance(trj, PaddedTrajectory):
        for i in range(trj.nframes):
            structure = Structure(
                cell=trj.cells[i].reshape([3, 3]),
                species=trj.symbols[i],
                positions=trj.positions[i].reshape([-1, 3]),
                forces=trj.forces[i].reshape([-1, 3]),
                energy=trj.energies[i],
            )
            structures += [structure]
    return structures


def write(filename, trj):

    structures = to_strucs(trj)
    with open(filename, "wb") as fout:
        pickle.dump(structures, fout)

    return structures


def from_file(filename, as_trajectory=True):
    per_frame_attrs = ["energies", "forces", "positions", "stresses"]
    mapping = dict(
        energy="energies",
        forces="forces",
        nat="natom",
        species_labels="species",
        _positions="positions",
        stress="stresses",
        _cell="cells",
    )
    structure_list = Structure.from_file(filename, as_trajectory=as_trajectory)
    trjs = Trajectories()
    for count, struc in enumerate(structure_list):
        d = struc.as_dict()
        new_dict = {mapping.get(k): np.array(d[k]) for k in mapping}
        new_dict["nframes"] = 1
        for key in per_frame_attrs:
            new_dict[key] = new_dict[key].reshape((1,) + new_dict[key].shape)
        trjs.add_trj(
            Trajectory.from_dict(
                new_dict,
                per_frame_attrs=[
                    "energies",
                    "forces",
                    "positions",
                    "stresses",
                    "cells",
                ],
            ),
            name=count,
        )
    return trjs.remerge()


import json
import numpy as np
from os.path import isfile
from math import inf

from flare.struc import Structure


def add_to_model(trj, gp_model, pre_train_env_per_species):

    test_structures = []
    count = 0

    for iframe in range(len(trj)):

        filename = filenames[itrj]
        data = np.load(filename)
        iskip = skip[itrj]

        meta = metas[itrj]
        # is_aimd = meta['aimd']

        cell = data["cell"]
        species = data["species"]
        positions = data["positions"]
        forces = data["forces"]
        energies = data["pe"]
        nframe = data["pe"].shape[0]
        natom = data["positions"].shape[1] // 3

        non_fix_atom = set(np.arange(natom))
        if meta.get("fix-atoms", True):
            begin = int(meta["non-fix-atoms"].split(".")[0])
            end = int(meta["non-fix-atoms"].split(".")[-1])
            non_fix_atom = set(np.arange(begin - 1, end))

        if iskip == 0:
            list_to_train = np.arange(0, nframe, 36)
        elif iskip == 1:
            list_to_train = np.arange(0, nframe, 2)

        for i in range(nframe):

            xyz = positions[i].reshape([-1, 3])
            structure = Structure(
                cell[i].reshape([3, 3]),
                species,
                xyz,
                forces=forces[i].reshape([-1, 3]),
                energy=energies[i],
            )

            if i in list_to_train:

                for species_i in set(structure.coded_species):

                    atoms_of_specie = set(structure.indices_of_specie(species_i))

                    add = False
                    if species_i in [6, 8]:
                        add = True
                    else:
                        if iskip == 0:
                            add = True
                        else:
                            if np.random.random() < 0.25:
                                add = True
                                list_to_add = []
                                for iatom in atoms_of_specie:
                                    if xyz[iatom, 2] > 12:
                                        list_to_add += [iatom]
                                atoms_of_specie = set(list_to_add)

                    if add:

                        # remove fixed/zero-forces atoms
                        list_to_add = list(non_fix_atom.intersection(atoms_of_specie))
                        np.random.shuffle(list_to_add)
                        n_at = len(list_to_add)

                        # Determine how many to add based on user defined cutoffs
                        n_to_add = min(
                            n_at, pre_train_env_per_species.get(species_i, inf), 2
                        )
                        if n_to_add > 0:
                            gp_model.update_db(
                                structure,
                                structure.forces,
                                custom_range=list_to_add[:n_to_add],
                            )
                            count += n_to_add

                        # print(f"Add {n_to_add} Atoms with specie {species_i} to model"
                        #       f" {gp_model.n_experts}; Total envs {count}")
                        print(
                            f"Add {n_to_add} Atoms with specie {species_i} to model"
                            f"; Total envs {count}",
                            iskip,
                            filename,
                        )
            else:
                test_structures += [structure]

    return test_structures
