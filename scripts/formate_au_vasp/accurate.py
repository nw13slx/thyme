from os.path import isfile

from thyme.filters.distance import e_filter
from thyme.filters.energy import sort_e
from thyme.routines.dist_plots.energy import multiple_plots as multiple_plots_e
from thyme.parsers.extxyz import write_trjs
from thyme.parsers.vasp import pack_folder_trj, get_childfolders, compare_metadata
from thyme.routines.folders import parse_merged_folders_trjs
from thyme.routines.folders import parse_folders_trjs
from thyme.trajectories import Trajectories
from ase.atoms import Atoms
import numpy as np
import logging

logging.basicConfig(
    filename=f"collect_merge.log",
    filemode="a",
    level=logging.INFO,
    format="%(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())


def main():

    folders = list(get_childfolders("../../../vasp/Au-formate"))
    for ch in [
        "recompute2",
        "recompute",
        "formate_Au110",
        "two_formate",
        "cross_edge",
        "melt_VASP",
        "pure_Au",
    ]:
        folders = folders + list(get_childfolders("../" + ch))

    if not isfile("all_data.pickle"):
        trjs = parse_merged_folders_trjs(
            folders,
            pack_folder_trj,
            e_filter,
            ckpt_filename="all_data.pickle",
            merge_level=1,
        )
    else:
        trjs = Trajectories.from_file("all_data.pickle")

    trjs = trjs.remerge(metadata_compare=compare_metadata, preserve_order=False)
    trjs.save("trjs_padded_mat.npz")
    multiple_plots_e(trjs, prefix="alldata")

    for name, trj in trjs.alldata.items():
        frames = sort_e(trj)
        trj.filter_frames(frames)
        logging.info(
            f"{repr(trj)} min and max E difference {np.max(trj.energies)-np.min(trj.energies)}"
        )
        mine = trj.energies[0]
        keep_id = np.where(trj.energies < (mine + 40))[0]
        trj.filter_frames(keep_id)
    trjs.save("trjs_remove_40up_padded_mat.npz")
    multiple_plots_e(trjs, prefix="40eV")
    write_trjs("all.xyz", trjs)


if __name__ == "__main__":
    main()
