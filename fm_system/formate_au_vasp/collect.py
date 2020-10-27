import logging
logging.basicConfig(filename=f'collect.log', filemode='w',
                                          level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

import numpy as np
from ase.atoms import Atoms

from fmeee.routines.folders import parse_folders_trjs
from fmeee.parsers.vasp import pack_folder_trj, get_childfolders
from fmeee.routines.dist_plots.energy import multiple_plots as multiple_plots_e


def main():

    folders = get_childfolders("./")
    trjs = parse_folders_trjs(folders, pack_folder_trj, e_filter, "all_data.pickle")
    trjs.save("alldata_padded_mat.npz")
    multiple_plots_e(trjs, prefix='alldata')

def e_filter(trj):
    accept_frame = []
    species = trj.species
    for i in range(trj.nframes):
        xyz = trj.positions[i]
        c = trj.cells[i]
        atoms = Atoms(species, xyz, cell=c, pbc=True)
        dist_mat = atoms.get_all_distances(mic=True)
        not_Au = [i for i in range(dist_mat.shape[0]) if species[i] != 'Au']
        for ind in range(dist_mat.shape[0]):

            neigh = np.argmin(np.hstack([dist_mat[ind, :ind], dist_mat[ind, ind+1:]]))
            if neigh >= ind:
                neigh += 1
            mindist = dist_mat[ind, neigh]

            if ind in not_Au and mindist > 2.0:
                logging.info(f"skip frame for isolated CHO atom {mindist} {ind}"
                             f" {species[ind]} {neigh} {species[neigh]}")
            elif ind not in not_Au:
                # if mindist > 3.0:
                #     logging.info(f"skip frame for isolated Au atom {mindist} {ind} {species[ind]} {neigh} {species[neigh]}")
                Au = [i for i in range(dist_mat.shape[0]) if species[i] == 'Au' and i!=ind]
                neigh = Au[np.argmin(dist_mat[ind, Au])]
                minAu = dist_mat[ind, neigh]
                if minAu > 3.5:
                    print(ind, dist_mat[ind, ind-3:np.min([ind+3, dist_mat.shape[0]])])
                    logging.info(f"skip frame for isolated Au from other Au {minAu} {ind}"
                                 f" {species[ind]} {neigh} {species[neigh]}")
                    accept_frame += [i]

    return np.array(accept_frame, dtype=int)

    # C_id = np.array([index for index, ele in enumerate(species) if ele == 'C'])
    # for Cindex in C_id:
    #     if xyz[Cindex, 2] < 10:
    #         logging.info("skip frame for low C")
    #         return False
    # if (e+411 < -100) or (e+411 > 40):
    #     logging.info(f"skip frame for high/low e {e+411:8.2f}")
    #     return False
    # return True

if __name__ == '__main__':
    main()
