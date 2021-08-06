import logging
import numpy as np
import pickle

from flare.struc import Structure

from thyme import Trajectory, Trajectories
from thyme._key import *


def to_strucs(trj):
    structures = []
    for i in range(trj.nframes):
        frame = trj.get_frame(i)
        structure = Structure(
            cell=frame[CELL],
            species=frame[SPECIES],
            positions=frame[POSITION],
            forces=frame[FORCE],
            energy=frame[TOTAL_ENERGY],
        )
        structures += [structure]
    return structures


def write(filename, trj):

    if hasattr(trj, "construct_id_list"):
        trj.construct_id_list(force_run=False)
    structures = to_strucs(trj)
    # with open(filename, "wb") as fout:
    #     pickle.dump(structures, fout)
    with open(filename, "w+") as fout:
        for struc in structures:
            print(struc.as_str(), file=fout)

    return structures


def from_file(filename, as_trajectory=True, **kwargs):
    """
    """
    mapping = dict(
        energy=TOTAL_ENERGY,
        forces=FORCE,
        species_labels=SPECIES,
        _positions=POSITION,
        stress=STRESS,
        _cell=CELL,
    )
    per_frame_attrs = [TOTAL_ENERGY, FORCE, POSITION, STRESS, CELL]

    structure_list = Structure.from_file(filename, as_trajectory=as_trajectory)
    trjs = Trajectories()
    for count, struc in enumerate(structure_list):
        d = struc.as_dict()
        new_dict = {mapping.get(k): np.array(d[k]) for k in mapping if k in d}
        for key in per_frame_attrs:
            new_dict[key] = new_dict[key].reshape((1,) + new_dict[key].shape)
        new_dict[PER_FRAME_ATTRS] = per_frame_attrs
        trjs.add_trj(
            Trajectory.from_dict(new_dict),
            name=count,
            **kwargs
        )
    return trjs


# import json
# import numpy as np
# from os.path import isfile
# from math import inf
#
# from flare.struc import Structure
#
#
# def add_to_model(trj, gp_model, pre_train_env_per_species):
#
#     test_structures = []
#     count = 0
#
#     for iframe in range(len(trj)):
#
#         filename = filenames[itrj]
#         data = np.load(filename)
#         iskip = skip[itrj]
#
#         meta = metas[itrj]
#         # is_aimd = meta['aimd']
#
#         cell = data["cell"]
#         species = data["species"]
#         positions = data["positions"]
#         forces = data["forces"]
#         total_energy = data["pe"]
#         nframe = data["pe"].shape[0]
#         natom = data["positions"].shape[1] // 3
#
#         non_fix_atom = set(np.arange(natom))
#         if meta.get("fix-atoms", True):
#             begin = int(meta["non-fix-atoms"].split(".")[0])
#             end = int(meta["non-fix-atoms"].split(".")[-1])
#             non_fix_atom = set(np.arange(begin - 1, end))
#
#         if iskip == 0:
#             list_to_train = np.arange(0, nframe, 36)
#         elif iskip == 1:
#             list_to_train = np.arange(0, nframe, 2)
#
#         for i in range(nframe):
#
#             xyz = positions[i].reshape([-1, 3])
#             structure = Structure(
#                 cell[i].reshape([3, 3]),
#                 species,
#                 xyz,
#                 forces=forces[i].reshape([-1, 3]),
#                 energy=total_energy[i],
#             )
#
#             if i in list_to_train:
#
#                 for species_i in set(structure.coded_species):
#
#                     atoms_of_specie = set(structure.indices_of_specie(species_i))
#
#                     add = False
#                     if species_i in [6, 8]:
#                         add = True
#                     else:
#                         if iskip == 0:
#                             add = True
#                         else:
#                             if np.random.random() < 0.25:
#                                 add = True
#                                 list_to_add = []
#                                 for iatom in atoms_of_specie:
#                                     if xyz[iatom, 2] > 12:
#                                         list_to_add += [iatom]
#                                 atoms_of_specie = set(list_to_add)
#
#                     if add:
#
#                         # remove fixed/zero-forces atoms
#                         list_to_add = list(non_fix_atom.intersection(atoms_of_specie))
#                         np.random.shuffle(list_to_add)
#                         n_at = len(list_to_add)
#
#                         # Determine how many to add based on user defined cutoffs
#                         n_to_add = min(
#                             n_at, pre_train_env_per_species.get(species_i, inf), 2
#                         )
#                         if n_to_add > 0:
#                             gp_model.update_db(
#                                 structure,
#                                 structure.forces,
#                                 custom_range=list_to_add[:n_to_add],
#                             )
#                             count += n_to_add
#
#                         # print(f"Add {n_to_add} Atoms with specie {species_i} to model"
#                         #       f" {gp_model.n_experts}; Total envs {count}")
#                         print(
#                             f"Add {n_to_add} Atoms with specie {species_i} to model"
#                             f"; Total envs {count}",
#                             iskip,
#                             filename,
#                         )
#             else:
#                 test_structures += [structure]
#
#     return test_structures
