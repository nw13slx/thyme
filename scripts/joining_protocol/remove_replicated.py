#!/bin/env python
import sys
from thyme.trajectories import Trajectories
from thyme.trajectory import Trajectory, PaddedTrajectory
from thyme.filters import rm_duplicate

files = sys.argv[1:-1]

alldata = PaddedTrajectory()
for filename in files:
    trjs = Trajectories.from_file(filename, format="padded_mat.npz")
    trj = trjs.to_padded_trajectory()
    print("add", trj)
    print("to", alldata)
    alldata.add_trj(trj)

keep_id = rm_duplicate(alldata)
alldata.include_frames(keep_id)

alldata.save(sys.argv[-1])
