import logging
logging.basicConfig(filename=f'plot_dist.log', filemode='w',
                                          level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

import numpy as np
from ase.atoms import Atoms

from fmeee.trajectories import Trajectories
from fmeee.parsers.cp2k import pack_folder_trj, get_childfolders
from fmeee.routines.dist_plots.energy import multiple_plots as multiple_plots_e
from fmeee.routines.dist_plots.energy import single_plot as single_plot_e
from fmeee.parsers.extxyz import write
from fmeee.filters.distance import e_filter

def main():

    trjs = Trajectories.from_file("all_data.pickle")
    logging.info("----FINAL TRJS----")
    logging.info(f"{trjs}")
    logging.info("-------END--------")

    # multiple_plots_e(trjs, prefix='folder')

    trjs = trjs.to_padded_trajectory()
    single_plot_e(trjs, prefix='merge')
    trjs = Trajectories.from_padded_trajectory(trjs,
                                               preserve_order=False)
    for i, trj in trjs.alldata.items():
        write(f"vmd{i}.xyz", trj)

    multiple_plots_e(trjs, prefix='elemenet')

    trjs.save("alldata_padded_mat.npz")

def e_filter(xyz, f, e, c, species):
    return True

if __name__ == '__main__':
    main()
