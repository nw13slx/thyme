from thyme.routines.parity_plots.base import base_parity
from thyme.routines.dist_plots.base import base_line_hist
from thyme.trajectories import Trajectories
from thyme.trajectory import PaddedTrajectory
from thyme.routines.write.xyz import write_single_xyz
from glob import glob
import sys
import numpy as np
import logging
logging.basicConfig(filename=f'filter.log', filemode='a',
                    level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

from ase.atoms import Atoms
from ase.io.extxyz import write_xyz as write_extxyz



# dictionary = dict(np.load(sys.argv[1], allow_pickle=True))

threshold = 0.2
all_trjs = []
min_length = -1
for filename in glob("result*.npz"):
    trjs = Trajectories.from_file(filename, format="padded_mat.npz")
    all_trjs += [trjs]

for spe_list in all_trjs[0].alldata:
    pred = []

    for i, trjs in enumerate(all_trjs):
        pred += [trjs.alldata[spe_list].pred.reshape([-1])]

    trj = all_trjs[0].alldata[spe_list]
    nframe = trj.pred.shape[0]
    Au_id = np.where(np.array(trj.species, dtype=str)=='Au')[0]

    pred = np.vstack(pred)
    pred_var = np.sqrt(np.var(pred, axis=0))

    # per atom max
    pred_var = pred_var.reshape([nframe, -1, 3])
    pred_var = np.max(pred_var, axis=-1)

    pred_Au = pred_var[:, Au_id]
    pred_Au_maxid = np.argmax(pred_Au, axis=1)

    pred_var = np.max(pred_var, axis=-1)
    exceed = len(np.where(pred_var > threshold)[0])
    logging.info(f"{exceed} out of {nframe} samples have error > {threshold}")

    sort_id = np.argsort(pred_var)
    for i in sort_id[-100:]:
        new_species = np.copy(trj.species)
        new_species[pred_Au_maxid[i]] = 'Cu'
        sort_spe = np.argsort(new_species)
        structure = Atoms(cell=trj.cells[i].reshape([3, 3]),
                          symbols=new_species[sort_spe],
                          positions=trj.positions[i].reshape([-1, 3])[sort_spe],
                          pbc=True)
        write_extxyz(f"highlight_{spe_list}.xyz", structure, append=True)

    trj.filter_frames(sort_id[-100:])
    write_single_xyz(trj, f"filter_{spe_list}")

# all_trjs[0].save("filtered.poscar")
# all_trjs[0].save("filtered_padded_mat.npz")
