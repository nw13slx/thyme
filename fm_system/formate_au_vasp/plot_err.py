import logging
logging.basicConfig(filename=f'plot.log', filemode='w',
                    level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())
import numpy as np
import sys

from fmeee.trajectories import Trajectories
from fmeee.routines.parity_plots.energy import multiple_plots as multiple_plots_e
from fmeee.routines.parity_plots.force import multiple_plots

# dictionary = dict(np.load(sys.argv[1], allow_pickle=True))

trjs = Trajectories.from_file(sys.argv[1])
trjs.save("trjs.pickle")

for trj in trjs.alldata.values():
    print(trj)

multiple_plots_e(trjs, pred_label='pe', prefix=sys.argv[2])
multiple_plots(trjs, pred_label='pred', prefix=sys.argv[2])
