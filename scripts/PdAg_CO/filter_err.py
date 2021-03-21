from thyme.routines.write.xyz import write_single_xyz
from thyme.filters.error import sort_by_force
from thyme.trajectories import Trajectories
import numpy as np
import logging

logging.basicConfig(
    filename=f"filter.log", filemode="a", level=logging.INFO, format="%(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())


skips = {
    "C1O1Pd7Ag144": [8, 10, 5],  # 5124 frames with 153 atoms
    "C1O1Pd5Ag144": [8, 10, 5],  # 171 frames with 151 atoms
    "C2O2Pd7Ag144": [4, 10, 5],  # 98 frames with 155 atoms
    "C2O2Pd7Ag143": [10, 10, 10],  # 2504 frames with 154 atoms
    "C3O3Pd7Ag143": [10, 10, 20],  # 2837 frames with 156 atoms
    "C4O4Pd7Ag144": [8, 10, 5],  # 290 frames with 159 atoms
    "C4O4Pd7Ag143": [8, 20, 1],  # 2117 frames with 158 atoms
    "Pd19Ag147": [60, 10, 1],  # 3536 frames with 166 atoms
    "Pd7Ag144": [0, 0, 0],  # 5 frames with 151 atoms
}


trjs = Trajectories.from_file("trjs.pickle")
train = Trajectories()
traindata = train.alldata
test = Trajectories()
testdata = test.alldata

for i, trj in enumerate(trjs.alldata.values()):
    if "O" not in trj.species:
        sorted_id = sort_by_force(trj, "pred", "Pd")
    else:
        sorted_id = sort_by_force(trj, "pred", "C")
    a = skips[trj.name]
    skip = a[0]
    cap = a[1]
    top = a[2]
    if skip > 0:
        new_trj = trj.skim(sorted_id[:-top:skip])
        new_trj2 = trj.skim(sorted_id[1:-top:skip][:cap])
        traindata[trj.name] = new_trj
        testdata[trj.name] = new_trj2

join_train = train.to_padded_trajectory()
join_train.shuffle()
join_test = test.to_padded_trajectory()
join_test.shuffle()

join_train.add_trj(join_test)

print(f"train {train.nframes}")
print(f"test {test.nframes}")
join_train.save(f"{train.nframes}frames_by_df.npz")
write_single_xyz(join_train, f"{train.nframes}frames_by_df")

# multiple_plots_e(trjs, pred_label='pe')
# multiple_plots(trjs, pred_label='pred')
