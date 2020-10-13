import logging
logging.basicConfig(filename=f'collect.log', filemode='w',
                    level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

import numpy as np

from ase.atoms import Atoms
from collections import Counter
from glob import glob
from os import walk, mkdir
from os.path import dirname, join, basename, isdir, isfile

from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar

from fmeee.vasp.parser import pack_folder
import fmeee.file_op import search_all_folders

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

def main():

    # mkfolder("npz")
    folders = search_all_folders(['vasprun.xml', 'OUTCAR', 'vasp.out'])
    folders = sorted(folders)
    logging.info(f"all folders: {folders}")

    count = 0
    alldata = {}
    for folder in folders:

        if folder == "./":
            casename = "current_folder"
        if folder[:2] == "./":
            casename = "_".join(folder[2:].split("/"))
        else:
            casename = "_".join(folder.split("/"))

        logging.info(casename)

        data = pack_folder(folder, e_filter)
        if data['nframes'] >= 1:
            logging.info(f"{folder}, {casename}, {data['nframes']}")
            alldata[casename] = data
            count += 1
            if count%10 == 0:
                np.savez("alldata.npz", **alldata)
        else:
            logging.info(f"! skip whole folder {casename}, {data['nframes']}")

    np.savez("alldata.npz", **alldata)
    logging.info("Complete")



if __name__ == '__main__':
    main()
