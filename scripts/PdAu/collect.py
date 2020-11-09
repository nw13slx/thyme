from thyme.parsers.vasp import pack_folder_trj, get_childfolders
from thyme.routines.dist_plots.force import multiple_plots as multiple_plots_f
from thyme.routines.dist_plots.energy import multiple_plots as multiple_plots_e
from thyme.routines.folders import parse_folders_trjs
from ase.atoms import Atoms
import numpy as np
import logging
logging.basicConfig(filename=f'collect.log', filemode='w',
                    level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())


def main():

    folders = get_childfolders("./")
    trjs = parse_folders_trjs(folders, pack_folder_trj,
                              e_filter, "all_data.pickle")
    trjs.save("alldata_padded_mat.npz")
    write("flare.pickle", trjs)
    multiple_plots_e(trjs, prefix='alldata')
    multiple_plots_f(trjs, prefix='alldata')


def e_filter(trj):

    return np.arange(trj.nframes)


if __name__ == '__main__':
    main()
