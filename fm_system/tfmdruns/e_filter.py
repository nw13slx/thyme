import matplotlib.pyplot as plt
from fmeee.routines.parity_plots.force import multiple_plots
from fmeee.routines.parity_plots.energy import multiple_plots as multiple_plots_e
from fmeee.routines.parity_plots.base import base_parity_ax
from fmeee.routines.dist_plots.base import base_line_hist
from fmeee.trajectories import Trajectories
from fmeee.trajectory import PaddedTrajectory
from fmeee.filters.error import sort_by_force
from glob import glob
import sys
import numpy as np
import logging
logging.basicConfig(filename=f'filter.log', filemode='a',
                    level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())
from fmeee.routines.write.xyz import write_single_xyz


filename = "ref1/result_3.npz"
trjs = Trajectories.from_file(
    filename, format="padded_mat.npz", preserve_order=False)
# original = Trajectories.from_file(
#     "1367_frames.npz", format="padded_mat.npz", preserve_order=False)
# ori_train = 1359

#original = Trajectories.from_file(
#    "1475_114.npz", format="padded_mat.npz", preserve_order=False)
#ori_train = 1475
#ori_test = 114

# original = Trajectories.from_file(
#     "827_frames.npz", format="padded_mat.npz", preserve_order=False)
# ori_train = 827
# ori_test = 10

original = Trajectories.from_file(
    "947_127.npz", format="padded_mat.npz", preserve_order=False)
ori_train = 947
ori_test = 127

nsample = 20

train = Trajectories()
test = Trajectories()
traindata = train.alldata
testdata = test.alldata

count = 0
count_test = 0
alle = []
for i, trj in enumerate(original.alldata.values()):
    if count < ori_train:
        select_id = np.arange(np.min([trj.nframes, ori_train-count]))
        if trj.nframes > (ori_train-count):
            if count_test < ori_test:
                test_id = np.arange(ori_train-count,
                                    np.min([trj.nframes,
                                            ori_test+ori_train-count_test-count]))
                new_trj = trj.skim(test_id)
                testdata[trj.name] = new_trj
                alle += [new_trj.energies]
                count_test += len(test_id)

        new_trj = trj.skim(select_id)
        traindata[trj.name] = new_trj
        alle += [new_trj.energies]
        count += len(select_id)
    elif count_test < ori_test:
        test_id = np.arange(np.min([trj.nframes,
                                    ori_test-count_test]))
        new_trj = trj.skim(test_id)
        testdata[trj.name] = new_trj
        alle += [new_trj.energies]
        count_test += len(test_id)

alle = np.hstack(alle)

for i, trj in enumerate(trjs.alldata.values()):
    if 'O' in trj.species:

        logging.info(f"{i} {trj}")

        sorted_id, err = sort_by_force(trj, 'pred', 'O')
        e_bar = np.where(trj.energies < (np.min(trj.energies)+60))[0]

        new_list = []
        for j in sorted_id:
            condi = len(np.where(trj.energies[j] == alle)[0])
            if condi > 0:
                logging.info(f"{trj.energies[j]} is seen before!")
            elif (err[j] > 0.05) and j in e_bar:
                new_list += [j]

        sorted_id = np.array(new_list, dtype=int)
        lower_bound = np.min([nsample, len(sorted_id)])
        train_id = sorted_id[-lower_bound:-2:3]
        test_id = sorted_id[-lower_bound+1:-2:3]

        logging.info(
            f"add to train: configs with err {err[train_id]}")
        logging.info(
            f"add to train: configs with energy {trj.energies[train_id]}")
        logging.info(
            f"add to test: configs with err {err[test_id]}")
        logging.info(
            f"add to test: configs with energy {trj.energies[test_id]}")

        new_trj = trj.skim(train_id)
        traindata[trj.name+"new"] = new_trj

        new_trj2 = trj.skim(test_id)
        testdata[trj.name+"new"] = new_trj2

        logging.info(
            f"{trj.name} select {new_trj.nframes} and {new_trj2.nframes}")

join_train = train.to_padded_trajectory()
join_train.shuffle()
join_test = test.to_padded_trajectory()
join_test.shuffle()

join_train.add_trj(join_test)

join_train.save(f"{train.nframes}_{test.nframes}.npz")
write_single_xyz(join_train, f"{train.nframes}_{test.nframes}")

# multiple_plots_e(trjs, pred_label='pe')
# multiple_plots(trjs, pred_label='pred')
