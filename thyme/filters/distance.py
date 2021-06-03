import logging
import numpy as np
from thyme._key import *

from ase.atoms import Atoms


def e_filter(trj, maxE=0, minE=-1e6):
    """
    filter away non-negative total_energy
    """

    accept_frame = []
    energy_id1 = np.where(trj.total_energy < maxE)[0]
    energy_id2 = np.where(trj.total_energy > minE)[0]
    energy_id = list(set(energy_id1).intersection(set(energy_id2)))
    skip_id = list(set(np.arange(len(trj.total_energy))) - set(energy_id))
    if len(skip_id) > 0:
        logging.info(f"skip frames for out of energy range [{minE},{maxE}] {skip_id}")

    for i in energy_id:
        frame = trj.get_frame(i, keys=[POSITION, CELL, SPECIES])
        species = frame[SPECIES]
        atoms = Atoms(species, frame[POSITION], cell=frame[CELL], pbc=True)
        dist_mat = atoms.get_all_distances(mic=True)
        not_metal = [
            i for i in range(dist_mat.shape[0]) if species[i] in ["C", "H", "O"]
        ]
        accept = True
        for ind in range(dist_mat.shape[0]):

            neigh = np.argmin(
                np.hstack([dist_mat[ind, :ind], dist_mat[ind, ind + 1 :]])
            )
            if neigh >= ind:
                neigh += 1
            mindist = dist_mat[ind, neigh]

            if ind in not_metal:
                if mindist > 2.0 and species[ind] != "O":
                    logging.info(
                        f"skip frame for isolated CHO atom {mindist} {ind}"
                        f" {species[ind]} {neigh} {species[neigh]}"
                    )
                    accept = False
                elif mindist > 3.0 and species[ind] == "O":
                    logging.info(
                        f"skip frame for isolated CHO atom {mindist} {ind}"
                        f" {species[ind]} {neigh} {species[neigh]}"
                    )
                    accept = False
            else:
                # if mindist > 3.0:
                #     logging.info(f"skip frame for isolated metal atom {mindist} {ind} {species[ind]} {neigh} {species[neigh]}")
                metal = [
                    i
                    for i in range(dist_mat.shape[0])
                    if species[i] not in ["C", "H", "O"] and i != ind
                ]
                neigh = metal[np.argmin(dist_mat[ind, metal])]
                min_metal = dist_mat[ind, neigh]
                if min_metal > 3.5:
                    print(
                        ind,
                        dist_mat[ind, ind - 3 : np.min([ind + 3, dist_mat.shape[0]])],
                    )
                    logging.info(
                        f"skip frame for isolated metal from other metal"
                        f" {min_metal} {ind}"
                        f" {species[ind]} {neigh} {species[neigh]}"
                    )
                    accept = False
        if accept:
            accept_frame += [i]

    return np.array(accept_frame, dtype=int)


def e_disp_filter(trj):
    """
    filter away non-negative total_energy
    and configs with isolated metal atoms
    """

    accept_frame = []
    species = trj.species
    energy_id = np.where(trj.total_energy < 0)[0]
    for i in energy_id:
        xyz = trj.position[i]
        c = trj.cell[i]
        atoms = Atoms(species, xyz, cell=c, pbc=True)
        dist_mat = atoms.get_all_distances(mic=True)
        not_metal = [
            i for i in range(dist_mat.shape[0]) if species[i] in ["C", "H", "O"]
        ]
        accept = True
        for ind in range(dist_mat.shape[0]):

            neigh = np.argmin(
                np.hstack([dist_mat[ind, :ind], dist_mat[ind, ind + 1 :]])
            )
            if neigh >= ind:
                neigh += 1
            mindist = dist_mat[ind, neigh]

            if ind in not_metal:
                if mindist > 2.0:
                    logging.info(
                        f"skip frame for isolated CHO atom {mindist} {ind}"
                        f" {species[ind]} {neigh} {species[neigh]}"
                    )
                    accept = False
            else:
                # if mindist > 3.0:
                #     logging.info(f"skip frame for isolated metal atom {mindist} {ind} {species[ind]} {neigh} {species[neigh]}")
                metal = [
                    i
                    for i in range(dist_mat.shape[0])
                    if species[i] not in ["C", "H", "O"] and i != ind
                ]
                neigh = metal[np.argmin(dist_mat[ind, metal])]
                min_metal = dist_mat[ind, neigh]
                if min_metal > 3.5:
                    print(
                        ind,
                        dist_mat[ind, ind - 3 : np.min([ind + 3, dist_mat.shape[0]])],
                    )
                    logging.info(
                        f"skip frame for isolated metal from other metal"
                        f" {min_metal} {ind}"
                        f" {species[ind]} {neigh} {species[neigh]}"
                    )
                    accept = False
        if accept:
            accept_frame += [i]

    return np.array(accept_frame, dtype=int)


def fixed_bottom(trj, layer_no=12):
    """
    filter away non-negative total_energy
    and configs with isolated metal atoms
    """

    accept_frame = []
    species = trj.species
    energy_id = np.where(trj.total_energy < 0)[0]

    for i in energy_id:

        if trj.is_padded:
            n = trj.natoms[i]
            xyz = trj.position[i][:n]
        else:
            xyz = trj.position[i]
        sort_id = np.argsort(xyz[:, 2])
        std_z = np.std(xyz[sort_id[:layer_no], 2])

        if std_z < 0.05:
            sf = np.sum(np.abs(trj.forces[i][sort_id[:layer_no]]))
            if sf != 0:
                accept_frame += [i]
            else:
                logging.info(f"skip config with 0 forces {sf}")
        else:
            logging.info(f"skip config with varying z {std_z}")

    return np.array(accept_frame, dtype=int)


# def mind_filter(trj):
#     """
#     filter away non-negative total_energy
#     and configs with isolated metal atoms
#     """
#
#     accept_frame = []
#     species = trj.species
#     energy_id = np.argsort(trj.total_energy)
#     for i in energy_id:
#         xyz = trj.position[i]
#         c = trj.cell[i]
#         atoms = Atoms(species, xyz, cell=c, pbc=True)
#         dist_mat = atoms.get_all_distances(mic=True)
#         not_metal = [i for i in range(dist_mat.shape[0]) if species[i] in ['C', 'H', 'O']]
#         accept = True
#         for ind in range(dist_mat.shape[0]):
#
#             neigh = np.argmin(np.hstack([dist_mat[ind, :ind], dist_mat[ind, ind+1:]]))
#             if neigh >= ind:
#                 neigh += 1
#             mindist = dist_mat[ind, neigh]
#
#             if ind in not_metal:
#                 if mindist > 2.0:
#                     logging.info(f"skip frame for isolated CHO atom {mindist} {ind}"
#                                  f" {species[ind]} {neigh} {species[neigh]}")
#                     accept = False
#             else:
#                 # if mindist > 3.0:
#                 #     logging.info(f"skip frame for isolated metal atom {mindist} {ind} {species[ind]} {neigh} {species[neigh]}")
#                 metal = [i for i in range(dist_mat.shape[0]) if species[i] not in \
#                       ['C', 'H', 'O'] and i!=ind]
#                 neigh = metal[np.argmin(dist_mat[ind, metal])]
#                 min_metal = dist_mat[ind, neigh]
#                 if min_metal > 3.5:
#                     print(ind, dist_mat[ind, ind-3:np.min([ind+3, dist_mat.shape[0]])])
#                     logging.info(f"skip frame for isolated metal from other metal"
#                                  f" {min_metal} {ind}"
#                                  f" {species[ind]} {neigh} {species[neigh]}")
#                     accept = False
#         if accept:
#             accept_frame += [i]
#
#     return np.array(accept_frame, dtype=int)
