import sys
from fmeee.trajectories import Trajectories
from fmeee.trajectory import Trajectory, PaddedTrajectory

file1 = sys.argv[1]
file2 = sys.argv[2]

types = [Trajectories, Trajectory, PaddedTrajectory]

for i in types:
    try:
        trjs = i.from_file(file1)
        print(f"work with type {i}")
    except:
        pass

trjs.save(file2)
