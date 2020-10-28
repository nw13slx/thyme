import logging
import numpy as np

from ase.atoms import Atoms

def e_filter(trj):
    """
    filter away non-negative energies
    and configs with isolated Au atoms
    """

    accept_frame = []
    species = trj.species
    energy_id = np.where(trj.energies<0)[0]
    for i in energy_id:
        xyz = trj.positions[i]
        c = trj.cells[i]
        atoms = Atoms(species, xyz, cell=c, pbc=True)
        dist_mat = atoms.get_all_distances(mic=True)
        not_Au = [i for i in range(dist_mat.shape[0]) if species[i] != 'Au']
        accept = True
        for ind in range(dist_mat.shape[0]):

            neigh = np.argmin(np.hstack([dist_mat[ind, :ind], dist_mat[ind, ind+1:]]))
            if neigh >= ind:
                neigh += 1
            mindist = dist_mat[ind, neigh]

            if ind in not_Au:
                if mindist > 2.0:
                    logging.info(f"skip frame for isolated CHO atom {mindist} {ind}"
                                 f" {species[ind]} {neigh} {species[neigh]}")
                    accept = False
            else:
                # if mindist > 3.0:
                #     logging.info(f"skip frame for isolated Au atom {mindist} {ind} {species[ind]} {neigh} {species[neigh]}")
                Au = [i for i in range(dist_mat.shape[0]) if species[i] == 'Au' and i!=ind]
                neigh = Au[np.argmin(dist_mat[ind, Au])]
                minAu = dist_mat[ind, neigh]
                if minAu > 3.5:
                    print(ind, dist_mat[ind, ind-3:np.min([ind+3, dist_mat.shape[0]])])
                    logging.info(f"skip frame for isolated Au from other Au {minAu} {ind}"
                                 f" {species[ind]} {neigh} {species[neigh]}")
                    accept = False
        if accept:
            accept_frame += [i]

    return np.array(accept_frame, dtype=int)
