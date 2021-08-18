import matplotlib.pyplot as plt
from thyme.routines.parity_plots.force import multiple_plots
from thyme.routines.parity_plots.energy import multiple_plots as multiple_plots_e
from thyme.routines.parity_plots.base import base_parity_ax
from thyme.routines.dist_plots.base import base_line_hist
from thyme.trajectories import Trajectories
from thyme.trajectory import PaddedTrajectory
from glob import glob
import sys
import numpy as np
import logging

logging.basicConfig(
    filename=f"plot.log", filemode="w", level=logging.INFO, format="%(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())


# dictionary = dict(np.load(sys.argv[1], allow_pickle=True))

upper_error = 1
grid = 1000

all_trjs = []
min_length = -1
for filename in glob("result*.npz"):
    trjs = Trajectories.from_file(
        filename, format="padded_mat.npz", preserve_order=True
    )
    all_trjs += [trjs]

for spe_list in all_trjs[0].alltrjs:

    # remove unphysicsl configs
    filter_id = np.where(trjs.alltrjs[spe_list].total_energy < 0)[0]

    fig_m, axs_m = plt.subplots(1, 2, figsize=(6.8, 2.5))
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))

    # collect committee variation
    pred = []
    for i, trjs in enumerate(all_trjs):
        trjs.alltrjs[spe_list].include_frames(filter_id)
        pred += [trjs.alltrjs[spe_list].pred.reshape([-1])]
        if i == 0:
            ref = trjs.alltrjs[spe_list].forces.reshape([-1])

    nframe = all_trjs[0].alltrjs[spe_list].pred.shape[0]
    skip = int(np.max([np.ceil(ref.shape[0] / 2000), 1]))
    pred = np.vstack(pred)
    pred_var = np.sqrt(np.var(pred, axis=0))
    pred_mean = np.average(pred, axis=0)

    # take the largest element out
    pred_var = pred_var.reshape([nframe, -1, 3])
    pred_var = np.max(pred_var, axis=-1)
    pred_var = pred_var.reshape([-1])

    base_line_hist(
        pred_var,
        "std (eV/A)",
        f"{spe_list}_ce_w_ref",
        lims=[None, 1],
        scatter_skip=skip,
    )
    for i, trjs in enumerate(all_trjs):
        error = np.abs(pred[i] - ref)
        error = error.reshape([nframe, -1, 3])
        error = np.max(error, axis=-1)
        error = error.reshape([-1])
        base_parity_ax(
            axs,
            error,
            pred_var,
            f"{spe_list}parity_e_ce",
            i,
            "True Error (eV/A)",
            "Predicted Error (eV/A)",
            shift=[],
            scatter_skip=skip,
        )
        x = []
        y = []
        for idx in range(grid):
            thred = (idx + 1) / grid * upper_error
            indices = np.where(error > thred)[0]
            if len(indices) > 0:
                x += [np.min(error[indices])]
                y += [np.min(pred_var[indices])]
        base_parity_ax(
            axs_m,
            np.array(x),
            np.array(y),
            f"{spe_list}parity_mine_mince",
            i,
            "Minimum True Error (eV/A)",
            "Minimum Predicted Error (eV/A)",
            shift=[],
            scatter_skip=skip,
        )
    fig_m.tight_layout()
    fig_m.savefig(f"{spe_list}parity_mine_mince.png", dpi=300)
    fig.tight_layout()
    fig.savefig(f"{spe_list}parity_e_ce.png", dpi=300)
    plt.close()
    del fig
    del axs
    del fig_m
    del axs_m

for i, trjs in enumerate(all_trjs):
    multiple_plots_e(trjs, pred_label="pe", prefix=f"{i}")
    multiple_plots(trjs, pred_label="pred", prefix=f"{i}")
