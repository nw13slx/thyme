from thyme.filters.energy import rm_duplicate
from thyme.routines.dist_plots.energy import single_plot as single_plot_e
from thyme.routines.dist_plots.energy import multiple_plots as multiple_plots_e
from thyme.parsers.cp2k import pack_folder_trj, get_childfolders
from thyme.trajectories import Trajectories
from thyme.parsers.extxyz import write
from ase.atoms import Atoms
import numpy as np
import logging

logging.basicConfig(
    filename=f"rm_fixed.log", filemode="w", level=logging.INFO, format="%(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())


def main():

    trjs1 = Trajectories.from_file("iterate/all_data.pickle")
    trjs2 = Trajectories.from_file("DIPOLE_CORRECTED/all_data.pickle")

    trjs = Trajectories()
    for i, trj in trjs1.alldata.items():
        fix = getattr(trj, "fix_atom", False)
        if not fix:
            trjs.alldata[f"iter_{i}"] = trj
        else:
            logging.info(f"remove fixed bottom {trj}")

    for i, trj in trjs2.alldata.items():
        fix = getattr(trj, "fix_atom", False)
        if not fix:
            trjs.alldata[f"ori_{i}"] = trj
        else:
            logging.info(f"remove fixed bottom {trj}")

    del trjs1
    del trjs2

    trjs = trjs.to_padded_trajectory()
    trjs = Trajectories.from_padded_trajectory(trjs, preserve_order=False)
    for i, trj in trjs.alldata.items():
        ids = rm_duplicate(trj)
        trj.include_frames(accept_id=ids)
        write(f"vmd{i}.xyz", trj)

    trjs.save("clean_up.padded_mat.npz")
    # logging.info("----FINAL TRJS----")
    # logging.info(f"{trjs}")
    # logging.info("-------END--------")

    trjs = Trajectories.from_file(
        "clean_up.padded_mat.npz", format="padded_mat.npz", preserve_order=False
    )


if __name__ == "__main__":
    main()
