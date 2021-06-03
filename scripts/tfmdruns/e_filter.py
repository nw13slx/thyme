from thyme.routines.write.xyz import write_single_xyz
import matplotlib.pyplot as plt
from thyme.routines.parity_plots.force import multiple_plots
from thyme.routines.parity_plots.energy import multiple_plots as multiple_plots_e
from thyme.routines.parity_plots.base import base_parity_ax
from thyme.routines.dist_plots.base import base_line_hist
from thyme.trajectories import Trajectories
from thyme.trajectory import PaddedTrajectory
from thyme.filters.error import sort_by_force
from glob import glob
import sys
import numpy as np
import logging

logging.basicConfig(
    filename=f"filter.log", filemode="a", level=logging.INFO, format="%(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())


# original = Trajectories.from_file(
#     "1367_frames.npz", format="padded_mat.npz", preserve_order=False)
# ori_train = 1359
# ori_test = 99

nsamples = [100, 20, 20]
# new_files = ["ref2/result_3.npz", "ref1/result_3.npz", "ref3/result_3.npz"]

threshold = 0.05

original = Trajectories.from_file(
    "827_frames.npz", format="padded_mat.npz", preserve_order=False
)
ori_train = 827
ori_test = 99
new_files = ["ref2/result_1.npz", "ref1/result_1.npz", "ref3/result_1.npz"]

train = Trajectories()
test = Trajectories()
traindata = train.alldata
testdata = test.alldata

count = 0
count_test = 0
alle = []
for i, trj in enumerate(original.alldata.values()):
    if count < ori_train:
        select_id = np.arange(np.min([trj.nframes, ori_train - count]))
        e_id = np.where(trj.total_energy > 0)[0]
        if len(e_id) > 0:
            logging.info(f" pop non-negative elements {e_id}")
        select_id = np.setdiff1d(select_id, e_id)
        # for idx in e_id:
        #     if idx in select_id:
        #         select_id.remove(idx)
        if trj.nframes > (ori_train - count):
            if count_test < ori_test:
                test_id = np.arange(
                    ori_train - count,
                    np.min([trj.nframes, ori_test + ori_train - count_test - count]),
                )
                e_id = np.where(trj.total_energy > 0)[0]
                if len(e_id) > 0:
                    logging.info(f" pop non-negative elements {e_id}")
                select_id = np.setdiff1d(select_id, e_id)
                # for idx in e_id:
                #     if idx in test_id:
                #         test_id.remove(idx)
                new_trj = trj.extract_frames(test_id)
                testdata[trj.name] = new_trj
                alle += [new_trj.total_energy]
                count_test += len(test_id)

        new_trj = trj.extract_frames(select_id)
        traindata[trj.name] = new_trj
        alle += [new_trj.total_energy]
        count += len(select_id)
    elif count_test < ori_test:
        test_id = np.arange(np.min([trj.nframes, ori_test - count_test]))
        new_trj = trj.extract_frames(test_id)
        testdata[trj.name] = new_trj
        alle += [new_trj.total_energy]
        count_test += len(test_id)

alle = np.hstack(alle)

for idf, filename in enumerate(new_files):

    trjs = Trajectories.from_file(
        filename, format="padded_mat.npz", preserve_order=False
    )
    nsample = nsamples[idf]

    for i, trj in enumerate(trjs.alldata.values()):
        if "O" in trj.species:

            logging.info(f"{i} {trj}")

            sorted_id, err = sort_by_force(trj, "pred", "O")

            # # remove top 3 highest energy configuration
            # e_id = np.argsort(trj.total_energy)[-3:]

            pop_id = [np.where(trj.total_energy > 0)[0]]
            pop_info = ["non-negative total_energy"]

            pop_id += [np.where(trj.total_energy > (np.min(trj.total_energy) + 60))[0]]
            pop_info += ["energy > 60 eV + min"]

            pop_id += [np.where(err < threshold)[0]]
            pop_info += [f"force error < {threshold}"]

            for ipop, pop in enumerate(pop_id):
                intersection = np.intersect1d(sorted_id, pop)
                if len(intersection) > 0:
                    logging.info(f"pop {pop_info[ipop]} {intersection}")
                sorted_id = np.setdiff1d(sorted_id, pop)

            # double check whether the config was included before
            new_list = []
            for j in sorted_id:
                condi = len(np.where(trj.total_energy[j] == alle)[0])
                if condi > 0:
                    logging.info(f"{trj.total_energy[j]} is seen before!")
                else:
                    new_list += [j]
                    alle = np.hstack([alle, [trj.total_energy[j]]])

            sorted_id = np.array(new_list, dtype=int)
            lower_bound = np.min([nsample, len(sorted_id)])
            train_id = sorted_id[-lower_bound::2]
            test_id = sorted_id[-lower_bound + 1 :: 2]

            new_trj = trj.extract_frames(train_id)
            if len(train_id) > 0:
                traindata[trj.name + "new"] = new_trj
                logging.info(f"add to train: configs with err {err[train_id]}")
                logging.info(
                    f"add to train: configs with energy {trj.total_energy[train_id]}"
                )

            new_trj2 = trj.extract_frames(test_id)
            if len(test_id) > 0:
                testdata[trj.name + "new"] = new_trj2
                logging.info(f"add to test: configs with err {err[test_id]}")
                logging.info(
                    f"add to test: configs with energy {trj.total_energy[test_id]}"
                )

            logging.info(f"{trj.name} select {new_trj.nframes} and {new_trj2.nframes}")

join_train = train.to_padded_trajectory()
join_train.shuffle()
join_test = test.to_padded_trajectory()
join_test.shuffle()

join_train.add_trj(join_test)

join_train.save(f"{train.nframes}_{test.nframes}.npz")
write_single_xyz(join_train, f"{train.nframes}_{test.nframes}")

# multiple_plots_e(trjs, pred_label='pe')
# multiple_plots(trjs, pred_label='pred')
