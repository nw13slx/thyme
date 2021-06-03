from thyme.routines.write.xyz import write_single_xyz
from thyme.filters.error import sort_by_force
from thyme.trajectories import Trajectories
import numpy as np
import logging

logging.basicConfig(
    filename=f"filter.log", filemode="w", level=logging.INFO, format="%(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())


trjs = Trajectories.from_file("trjs.pickle")
train = Trajectories()
traindata = train.alldata
test = Trajectories()
testdata = test.alldata

for i, trj in enumerate(trjs.alldata.values()):
    sorted_id = sort_by_force(trj, "pred", "O")
    if i == 0:
        new_trj = trj.extract_frames(sorted_id[:-20:3])
        new_trj2 = trj.extract_frames(sorted_id[1:-20:3])
    else:
        new_trj = trj.extract_frames(sorted_id[:-50:30])
        new_trj2 = trj.extract_frames(sorted_id[1:-50:30])
    traindata[trj.name] = new_trj
    testdata[trj.name] = new_trj2

join_train = train.to_padded_trajectory()
join_train.shuffle()
join_test = test.to_padded_trajectory()
join_test.shuffle()

join_train.add_trj(join_test)

join_train.save(f"{train.nframes}frames_by_df.npz")
write_single_xyz(join_train, f"{train.nframes}frames_by_df")

# multiple_plots_e(trjs, pred_label='pe')
# multiple_plots(trjs, pred_label='pred')
