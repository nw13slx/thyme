from thyme.filters.distance import e_filter
from thyme.routines.dist_plots.energy import multiple_plots as multiple_plots_e
from thyme.parsers.cp2k import pack_folder_trj, get_childfolders
from thyme.routines.folders import parse_merged_folders_trjs
from ase.atoms import Atoms
import numpy as np
import logging

logging.basicConfig(
    filename=f"collect.log", filemode="w", level=logging.INFO, format="%(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())


def main():

    folders = get_childfolders("./")
    trjs = parse_merged_folders_trjs(
        folders,
        pack_folder_trj,
        data_filter=e_filter,
        ckpt_filename="all_data.pickle",
        merge_level=0,
    )
    logging.info("----FINAL TRJS----")
    logging.info(f"{trjs}")
    logging.info("-------END--------")
    trjs.save("alldata.npz")


def e_filter(xyz, f, e, c, species):
    return True


if __name__ == "__main__":
    main()
