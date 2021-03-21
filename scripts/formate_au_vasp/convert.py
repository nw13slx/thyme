from thyme.routines.parity_plots.force import multiple_plots
from thyme.routines.parity_plots.energy import multiple_plots as multiple_plots_e
from thyme.trajectories import Trajectories
import numpy as np
import logging

logging.basicConfig(
    filename=f"convert.log", filemode="w", level=logging.INFO, format="%(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())

dictionary = dict(np.load("alldata.npz", allow_pickle=True))

trjs = Trajectories.from_dict(dictionary, merge=True)
trjs.save("trjs.pickle")
trjs.save("trjs_padded_mat.npz")
