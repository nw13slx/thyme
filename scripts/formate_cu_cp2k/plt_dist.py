from thyme.filters.distance import e_filter
from thyme.routines.dist_plots.energy import single_plot as single_plot_e
from thyme.routines.dist_plots.energy import multiple_plots as multiple_plots_e
from thyme.parsers.cp2k import pack_folder_trj, get_childfolders
from thyme.trajectories import Trajectories
from thyme.parsers.extxyz import write
from ase.atoms import Atoms
import numpy as np
import logging

logging.basicConfig(
    filename=f"plot_dist.log", filemode="w", level=logging.INFO, format="%(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())


def main():

    trjs = Trajectories.from_file("all_data.pickle")
    logging.info("----FINAL TRJS----")
    logging.info(f"{trjs}")
    logging.info("-------END--------")

    # multiple_plots_e(trjs, prefix='folder')

    trjs = trjs.to_padded_trajectory()
    single_plot_e(trjs, prefix="merge")
    trjs = Trajectories.from_padded_trajectory(trjs, preserve_order=False)
    for i, trj in trjs.alltrjs.items():
        write(f"vmd{i}.xyz", trj)

    multiple_plots_e(trjs, prefix="elemenet")

    trjs.save("alldata_padded_mat.npz")


def e_filter(xyz, f, e, c, species):
    return True


if __name__ == "__main__":
    main()
