from thyme.routines.parity_plots.force import multiple_plots
from thyme.routines.parity_plots.energy import multiple_plots as multiple_plots_e
from thyme.trajectories import Trajectories
import numpy as np
import logging

logging.basicConfig(
    filename=f"plot.log", filemode="w", level=logging.INFO, format="%(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())

dictionary = dict(np.load("all_results.npz", allow_pickle=True))

trjs = Trajectories.from_padded_matrices(dictionary)
trjs.save("trjs.pickle")

for trj in trjs.alldata.values():
    print(trj)

multiple_plots_e(trjs, pred_label="pe")
multiple_plots(trjs, pred_label="pred")
