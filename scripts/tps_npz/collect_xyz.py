from thyme.routines.dist_plots.energy import single_plot as single_plot_e
from thyme.parsers.pysampling_xyz import pack_folder_trj, get_childfolders
from thyme.routines.folders import parse_folders_trjs
from ase.atoms import Atoms
from collections import Counter
import numpy as np
import logging
logging.basicConfig(filename=f'collect.log', filemode='w',
                    level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

def main():

    folders = get_childfolders("./", include_xyz=True)
    trjs = parse_folders_trjs(folders, pack_folder_trj,
                              None, "results.pickle")

    for i, trj in trjs.alldata.items():

        labeling(trj)

        for iconfig in range(trj.nframes):

            # shift with H
            hpos = trj.positions[iconfig][0, :2]
            trj.positions[iconfig][:, :2] -= hpos

            # wrap around
            for idir in range(2):
                trj.positions[iconfig][:, idir] += trj.cells[iconfig][idir, idir]
                trj.positions[iconfig][:, idir] = \
                    trj.positions[iconfig][:, idir]%trj.cells[iconfig][idir, idir]

    trj = trjs.to_padded_trajectory()
    c = Counter(trj.labels)
    trj.save("results_padded_mat.npz")
    trj.name = 'allresults'
    single_plot_e(trj, label='labels', prefix='', xlabel='Basin Label')

    logging.info(f"overall label {c}")

    for label in [-1, 0, 1]:
        sort_id = np.where(trj.labels==label)[0]
        new_trj = trj.skim(sort_id)
        new_trj.save(f"l{label}results_padded_mat.npz")
        new_trj.name = f'l{label}'
        single_plot_e(new_trj, prefix=f'')


def labeling(trj):

    species = trj.symbols[-1]
    atoms = Atoms(species, trj.positions[-1], cell=trj.cells[-1], pbc=True)

    Hid = np.array([i for i, s in enumerate(species) if s == 'H'])
    Cid = np.array([i for i, s in enumerate(species) if s == 'C'])
    Oid = np.array([i for i, s in enumerate(species) if s == 'O'])
    Cuid = np.array([i for i, s in enumerate(species) if s == 'Cu'])

    dist_mat = atoms.get_all_distances(mic=True)
    dist_CH = np.max((dist_mat[Hid].T)[Cid])

    all_OCu = np.min((dist_mat[Oid].T)[Cuid], axis=0)
    dist_OCu_min = np.min(all_OCu)
    dist_OCu_max = np.max(all_OCu)
    dist_HCu = np.min((dist_mat[Hid].T)[Cuid])

    label = -1
    if dist_CH > 2.5 and dist_OCu_max > 3 and dist_HCu < 1.8:
        label = 2
    elif dist_CH <= 1.2 and dist_OCu_max < 2.5 and dist_HCu > 2.5:
        label = 0
    elif dist_CH <= 1.2 and dist_OCu_min < 2.5:
        label = 1
        logging.info(f"find monodentate {dist_CH:5.2f} "
                     f"OCu {dist_OCu_max:5.2f} {dist_OCu_min:5.2f} HCu {dist_HCu:5.2f}")

    trj.labels = np.zeros(trj.nframes) + label
    trj.per_frame_attrs += ['labels']
    logging.info(f"label as {label}")
    if label == -1:
        logging.info(f"fail to find basin CH {dist_CH:5.2f} "
                     f"OCu {dist_OCu_max:5.2f} {dist_OCu_min:5.2f} HCu {dist_HCu:5.2f}")

if __name__ == '__main__':
    main()
