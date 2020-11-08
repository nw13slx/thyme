import logging
logging.basicConfig(filename=f'convert.log', filemode='w',
                    level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())
import numpy as np
from thyme.trajectories import Trajectories
from thyme.routines.parity_plots.energy import multiple_plots as multiple_plots_e
from thyme.routines.parity_plots.force import multiple_plots

dictionary = dict(np.load("alldata.npz", allow_pickle=True))

trjs = Trajectories.from_dict(dictionary, merge=True)
trjs.save("trjs.pickle")
trjs.save("trjs_padded_mat.npz")
