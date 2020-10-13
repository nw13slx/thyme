import logging
logging.basicConfig(filename=f'collect.log', filemode='w',
                                          level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

import numpy as np

from fmeee.routines.folders import parse_folders
from fmeee.parsers.vasp import pack_folder, get_childfolders

from pymatgen.core.structure import Structure


def main():

    folders = get_childfolders("./")
    parse_folders(folders, pack_folder, o_dist_filter, "all_data")

def o_dist_filter(o_xyz, f, e, c, species, mine=-411, ne=2):

    xyz = o_xyz.reshape([-1, 3])
    atoms =  Structure(lattice=c,
                     species=species,
                     coords=xyz,
                     coords_are_cartesian=True)
    dist_mat = atoms.distance_matrix

    C_id = np.array([index for index, ele in enumerate(species) if ele == 'C'])
    O_id = np.array([index for index, ele in enumerate(species) if ele == 'O'])

    nPd = len([ele for ele in species if ele == 'Pd'])
    nAg = len([ele for ele in species if ele == 'Ag'])
    nCO = len([ele for ele in species if ele == 'C'])

    refe = -(nPd*5.1765+nAg*2.8325+nCO*1.8526)
    # de = e-refe
    if ne > 5:
        de = e-mine
    else:
        de = e-refe

    for Cindex in C_id:
        dist_C = np.delete(dist_mat[Cindex, :], Cindex)
        dist_C = np.min(dist_C)
        if dist_C > 1.7:
            logging.info(f"skip frame for isolated C {dist_C:5.1f} "
                         f"{e-mine:6.2f}+{mine:6.2f}"
                         f"{e-refe:6.2f}+{refe:6.2f}")
            return False
        if xyz[Cindex, 2] < 10:
            logging.info(f"skip frame for low C {xyz[Cindex, 2]:5.1f}"
                         f"{e-mine:6.2f}+{mine:6.2f}"
                         f"{e-refe:6.2f}+{refe:6.2f}")
            return False
    for Oindex in O_id:
        dist_O = np.delete(dist_mat[Oindex, :], Oindex)
        dist_O = np.min(dist_O)
        if dist_O > 1.7:
            logging.info(f"skip frame for isolated O {dist_O:5.1f}"
                         f"{e-mine:6.2f}+{mine:6.2f}"
                         f"{e-refe:6.2f}+{refe:6.2f}")
            return False
    if (de < -100) or (de > 100):
        logging.info(f"skip frame for high/low "
                         f"{e-mine:6.2f}+{mine:6.2f}"
                         f"{e-refe:6.2f}+{refe:6.2f}")
        return False

    return True

if __name__ == '__main__':
    main()
