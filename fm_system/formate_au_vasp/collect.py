import logging
logging.basicConfig(filename=f'collect.log', filemode='w',
                                          level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

import numpy as np
from ase.atoms import Atoms

from fmeee.routines.folders import parse_folders
from fmeee.parsers.vasp import pack_folder, get_childfolders

def main():

    folders = get_childfolders("./")
    parse_folders(folders, pack_folder, e_filter, "all_data")

def e_filter(xyz, f, e, c, species):
    atoms = Atoms(species, xyz, cell=c, pbc=True)
    dist_mat = atoms.get_all_distances(mic=True)
    not_Au = [i for i in range(dist_mat.shape[0]) if species[i] != 'Au']
    for ind in range(dist_mat.shape[0]):

        neigh = np.argmin(np.hstack([dist_mat[ind, :ind], dist_mat[ind, ind+1:]]))
        if neigh >= ind:
            neigh += 1
        mindist = dist_mat[ind, neigh]

        if ind in not_Au and mindist > 2.0:
            logging.info(f"skip frame for isolated CHO atom {mindist} {ind} {species[ind]} {neigh} {species[neigh]}")
            return False
        elif ind not in not_Au:
            # if mindist > 3.0:
            #     logging.info(f"skip frame for isolated Au atom {mindist} {ind} {species[ind]} {neigh} {species[neigh]}")
            #     return False
            Au = [i for i in range(dist_mat.shape[0]) if species[i] == 'Au' and i!=ind]
            neigh = Au[np.argmin(dist_mat[ind, Au])]
            minAu = dist_mat[ind, neigh]
            if minAu > 3.5:
                print(ind, dist_mat[ind, ind-3:np.min([ind+3, dist_mat.shape[0]])])
                logging.info(f"skip frame for isolated Au from other Au {minAu} {ind} {species[ind]} {neigh} {species[neigh]}")
                return False


    return True
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
