import logging
logging.basicConfig(filename=f'filter.log', filemode='w',
                    level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

import numpy as np

from fmeee.trajectories import Trajectories
from fmeee.filters.error import sort_by_force


trjs = Trajectories.from_file("trjs.pickle")
new_trjs = Trajectories()
alldata = new_trjs.alldata

for i, trj in enumerate(trjs.alldata.values()):
    sorted_id = sort_by_force(trj, 'pred', 'O')
    if i==0:
        new_trj = trj.skim(sorted_id[:-10:3])
    else:
        new_trj = trj.skim(sorted_id[:-20:50])
    alldata[trj.name] = new_trj
    print(new_trjs.nframes)
new_trjs.save_padded_matrices(f"{new_trjs.nframes}frames_by_df.npz")

