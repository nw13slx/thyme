from thyme.routines.parity_plots.base import base_parity
from thyme.routines.dist_plots.base import base_line_hist_axs
from thyme.trajectories import Trajectories
from thyme.trajectory import PaddedTrajectory
from glob import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    filename=f"plot.log", filemode="w", level=logging.INFO, format="%(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())

plt.switch_backend("agg")


# dictionary = dict(np.load(sys.argv[1], allow_pickle=True))

all_trjs = []
min_length = -1
for filename in glob("result*.npz"):
    trjs = Trajectories.from_file(
        filename, format="padded_mat.npz", preserve_order=True
    )
    all_trjs += [trjs]


for spe_list in all_trjs[0].alltrjs:
    pred = []
    nframe = all_trjs[0].alltrjs[spe_list].pred.shape[0]
    species = np.array(all_trjs[0].alltrjs[spe_list].species, dtype=str)
    for i, trjs in enumerate(all_trjs):
        pred += [trjs.alltrjs[spe_list].pred.reshape([-1])]

    pred = np.vstack(pred)
    pred_var = np.sqrt(np.var(pred, axis=0))

    # take the largest element out
    pred_var = pred_var.reshape([nframe, -1, 3])
    all_pred_var = np.max(pred_var, axis=-1)

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for ele in set(species):
        symbol_id = np.where(species == ele)[0]
        pred_var = np.max(all_pred_var[:, symbol_id], axis=-1)
        skip = int(np.max([np.ceil(pred_var.shape[0] / 2000), 1]))
        logging.info(f"{spe_list} {ele} skip {skip}")
        base_line_hist_axs(
            axs,
            pred_var,
            "std (eV/A)",
            ele,
            f"{spe_list}_ce_wo_ref",
            lims=[0, 2],
            scatter_skip=skip,
        )
    fig.tight_layout()
    fig.savefig(f"{spe_list}_ce_wo_ref.png", dpi=300)
    plt.close()
