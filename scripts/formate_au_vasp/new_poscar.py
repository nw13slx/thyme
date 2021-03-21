from thyme.trajectories import Trajectories
from thyme.trajectory import Trajectory
from thyme.filters.distance import e_filter
from thyme.filters.energy import sort_e
from thyme.routines.dist_plots.energy import multiple_plots as multiple_plots_e
from thyme.parsers.vasp import pack_folder_trj, get_childfolders, write
from thyme.routines.folders import parse_folders_trjs
from ase.atoms import Atoms
import numpy as np
import logging

logging.basicConfig(
    filename=f"new_poscar.log", filemode="w", level=logging.DEBUG, format="%(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())


def main():

    trjs = Trajectories.from_file("all_data.pickle")
    trjs = trjs.remerge(preserve_order=False)
    trjs.save("all_data_merged.pickle")

    mineT = Trajectory()
    for name, trj in trjs.alldata.items():
        mine = np.min(trj.energies)
        keep_id = np.where(trj.energies < (mine + 20))[0]
        trj.filter_frames(keep_id)
        write("result_pos/"+name, trj)

        print(name, trj.energies)
        # trj.filter_frames(keep_id)


if __name__ == "__main__":
    main()
