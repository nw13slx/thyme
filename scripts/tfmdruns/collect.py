from thyme.routines.dist_plots.energy import multiple_plots as multiple_plots_e
from thyme.parsers.tfmd import pack_folder_trj, get_childfolders
from thyme.routines.folders import parse_folders_trjs
from ase.atoms import Atoms
import numpy as np
import logging

logging.basicConfig(
    filename=f"collect.log", filemode="w", level=logging.INFO, format="%(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())


def main():

    folders = get_childfolders("./", include_xyz=True)
    trjs = parse_folders_trjs(folders, pack_folder_trj, e_filter, "all_data.pickle")
    trjs.save("alldata_padded_mat.npz")
    multiple_plots_e(trjs, prefix="alldata")


def e_filter(trj):
    set1 = set(np.where(trj.total_energy < 0)[0])
    set2 = rm_sudden_drop(trj, 10)

    # set2 = set(np.where(trj.total_energy > -3000)[0])
    return list(set1.intersection(set2))


if __name__ == "__main__":
    main()
