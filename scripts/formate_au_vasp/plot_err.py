from thyme.routines.parity_plots.force import multiple_plots
from thyme.routines.parity_plots.energy import multiple_plots as multiple_plots_e
from thyme.trajectories import Trajectories
import sys
import numpy as np
import logging

logging.basicConfig(
    filename=f"plot.log", filemode="w", level=logging.INFO, format="%(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())


# dictionary = dict(np.load(sys.argv[1], allow_pickle=True))

trjs = Trajectories.from_file(sys.argv[1], format="padded_mat.npz")
trjs.save("trjs.pickle")

for trj in trjs.alltrjs.values():
    print(trj)

multiple_plots_e(trjs, pred_label="pe", prefix=sys.argv[2])
multiple_plots(trjs, pred_label="pred", prefix=sys.argv[2])
