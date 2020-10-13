import json
import logging
import numpy as np

from numpy.random import shuffle
from collections import Counter
from glob import glob
from os import walk, mkdir
from os.path import dirname, join, basename, isdir, isfile
import sys

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar

folder = dirname(sys.argv[1])
name = "_".join(sys.argv[2:])
logging.basicConfig(filename=f'{folder}/split_{name}.log', filemode='w',
                    level=logging.DEBUG, format="%(message)s")

def o_dist_filter(o_xyz, f, e, c, species):
    return True

    xyz = o_xyz.reshape([-1, 3])
    atoms =  Structure(lattice=c,
                     species=species,
                     coords=xyz,
                     coords_are_cartesian=True)
    dist_mat = atoms.distance_matrix

    C_id = np.array([index for index, ele in enumerate(species) if ele == 'C'])
    O_id = np.array([index for index, ele in enumerate(species) if ele == 'O'])

    nPd = len([ele for ele in species if ele == 'Pd'])
    nAg = len([ele for ele in species if ele == 'Ag'])
    nCO = len([ele for ele in species if ele == 'C'])


    for Cindex in C_id:
        if xyz[Cindex, 2] < 10:
            logging.info(f"skip frame for low C {xyz[Cindex, 2]:5.1f}")
            return False
        dist_C = np.delete(dist_mat[Cindex, :], Cindex)
        dist_C = np.min(dist_C)
        if dist_C > 1.7:
            logging.info(f"skip frame for isolated C {dist_C:5.1f} {e+411:6.2f}")
            return False
    refe = -(nPd*5.1765+nAg*2.8325+nCO*1.8526)
    de = e-refe
    if (de < -100) or (de > 10):
        logging.info(f"skip frame for high/low e-refe {de:6.2f}")
        return False
    for Oindex in O_id:
        dist_O = np.delete(dist_mat[Oindex, :], Oindex)
        dist_O = np.min(dist_O)
        if dist_O > 1.7:
            logging.info(f"skip frame for isolated O {dist_O:5.1f} {e+411:6.2f}")
            return False

    return True

def main():

    if len(sys.argv)<3:
        logging.info("python analysis/split_trj.py filename ntrain nvalid ntest whole md")
        return

    alldata = dict(np.load(sys.argv[1], allow_pickle=True))
    data_filter = o_dist_filter

    trjnames = sort_filenames(alldata)
    shuffle(trjnames)

    ntrain = int(sys.argv[2])
    nvalid = int(sys.argv[3])
    ntest = int(sys.argv[4])

    whole = bool(sys.argv[5])
    md = int(sys.argv[6])
    if len(sys.argv) > 7:
        do_shuffle = bool(sys.argv[7])
    else:
        do_shuffle = False
    if len(sys.argv) > 8:
        skip = int(sys.argv[8])
    else:
        skip = 1

    positions = {}
    forces = {}
    energies = {}
    cells = {}
    count = {}
    tasks = ['train', 'valid', 'test', 'spare']
    list_of_trj = {}
    for task in tasks:
        positions[task] = []
        forces[task] = []
        energies[task] = []
        cells[task] = []
        count[task] = 0
        list_of_trj[task] = []

    nframes = 0
    ntrjs = 0

    if md==2:

        total_frames = 0
        for i, trjname in enumerate(trjnames):
            data = alldata[trjname].item()
            frame = data['positions'].shape[0]
            total_frames += len(np.arange(0, frame, skip))


    for i, trjname in enumerate(trjnames):

        data = alldata[trjname].item()
        frame = data['positions'].shape[0]
        logging.debug(f"before filter, {frame}")
        data = filter_trj(data, data_filter)
        frame = data['nframes']
        logging.debug(f"after filter, {frame}")

        if frame > 0:

            list_to = {}

            if md==0:

                id_list = np.arange(0, frame, skip)
                if do_shuffle:
                    shuffle(id_list)

                frame = len(id_list)

                if count['train'] < ntrain and (frame > 1 or whole):
                    list_to['train'] = np.arange(0, frame)
                elif count['valid'] < nvalid:
                    list_to['valid'] = np.arange(0, frame)
                elif count['test'] < ntest:
                    list_to['test'] = np.arange(0, frame)
                else:
                    list_to['spare'] = np.arange(0, frame)

            elif md==2:

                id_list = np.arange(0, frame, skip)
                skip_frame = len(id_list)

                this_train = int(np.floor(ntrain/total_frames*skip_frame))
                this_valid = np.min((int(np.ceil(nvalid/total_frames*skip_frame)), skip_frame-this_train))
                this_test = np.min((int(np.ceil(ntest/total_frames*skip_frame)), skip_frame-this_valid-this_train))
                this_spare = np.max((skip_frame - this_test - this_valid - this_train, 0))
                print(trjname, skip_frame, this_train, this_valid, this_test, this_spare)

                if do_shuffle:
                    id_train_valid = id_list[:this_train+this_valid]
                    id_rest = id_list[this_train+this_valid:]
                    shuffle(id_train_valid)
                    id_list = np.hstack((id_train_valid, id_rest))

                if this_train > 0:
                    list_to['train'] = id_list[:this_train]
                if this_valid > 0:
                    list_to['valid'] = id_list[this_train:this_train+this_valid]
                if this_test > 0:
                    list_to['test'] = id_list[this_train+this_valid:this_test+this_train+this_valid]
                if this_spare > 0:
                    list_to['spare'] = id_list[skip_frame-this_spare:]

            elif md==1:

                if count['train'] < ntrain and (frame > 1 or whole):

                    end_train = np.min([frame, ntrain-count['train']])
                    end_val = np.min([frame-end_train, nvalid-count['valid']])

                    shuffle_id = np.random.permutation(end_train+end_val)

                    list_to['train'] = shuffle_id[0:end_train]
                    if end_val > 0:
                        list_to['valid'] = shuffle_id[end_train:]

                    remain = end_train+end_val
                    if end_train+end_val < frame:
                        end_test = np.min([frame-remain, ntest-count['test']])
                        if end_test > 0:
                            list_to['test'] = np.arange(remain, end_test+remain)
                        remain += end_test
                        if remain < frame:
                            list_to['spare'] = np.arange(remain, frame)

                elif count['valid'] < nvalid:

                    end_val = np.min([frame, nvalid-count['valid']])

                    list_to['valid'] = np.arange(0, end_val)

                    remain = end_val
                    if end_val < frame:
                        end_test = np.min([frame-remain, ntest-count['test']])
                        if end_test > 0:
                            list_to['test'] = np.arange(remain, end_test+remain)
                        remain += end_test
                        if remain < frame:
                            list_to['spare'] = np.arange(remain, frame)

                elif count['test'] < ntest:

                    end_test = np.min([frame, ntest-count['test']])
                    list_to['test'] = np.arange(0, end_test)

                    if end_test < frame:
                        list_to['spare'] = np.arange(end_test, frame)
                else:
                    list_to['spare'] = np.arange(0, frame)

                if do_shuffle:
                    shuffle_id = np.random.permutation(frame)

            for task in list_to:


                this_frame = len(list_to[task])

                logging.info(f"{trjname} {task} {this_frame}")

                if do_shuffle:
                    ltt = shuffle_id[list_to[task]]
                else:
                    ltt = list_to[task]

                count[task] += this_frame
                list_of_trj[task] += [(trjname, this_frame)]
                positions[task] += [data['positions'][ltt]]
                forces[task] += [data['forces'][ltt]]
                energies[task] += [data['energies'][ltt]]
                cells[task] += [data['cells'][ltt]]


    for task in tasks:
        if len(positions[task]) > 0:
            positions[task] = np.vstack(positions[task])
            forces[task] = np.vstack(forces[task])
            energies[task] = np.hstack(energies[task])
            cells[task] = np.vstack(cells[task])
            # logging.info(positions.shape, forces.shape,
            #       energies.shape, cells.shape)
            np.savez(f"{folder}/{task}.npz", positions=positions[task],
                     forces=forces[task], energies=energies[task],
                     cells=cells[task])
    with open(f"{folder}/split_details.json", "w+") as fout:
        json.dump(list_of_trj, fout)

    logging.info(f"{folder} {count}")
    return 0

def sort_filenames(data):

    filenames = []
    file_indices = []
    for trjname in data:
        try:
            trj_index = int(re.sub('[^\d]', '', trjname))
        except Exception as e:
            # logging.info("failed", e)
            trj_index = -1
        filenames += [trjname]
        file_indices += [trj_index]
    filenames = np.array(filenames)
    file_indices = np.argsort(np.array(file_indices))
    return filenames[file_indices]


def filter_trj(data, data_filter):

    xyzs = data['positions']
    fs = data['forces']
    es = data['energies']
    cs = data['cells']
    nframes = xyzs.shape[0]

    species = data['species']
    positions = []
    forces = []
    energies = []
    cells = []
    for istep in range(nframes):
        xyz = xyzs[istep]
        f = fs[istep]
        e = es[istep]
        c = cs[istep]
        if data_filter(xyz, f, e, c, species):
            positions += [xyz.reshape([-1])]
            forces += [np.hstack(f)]
            energies += [e]
            cells += [c.reshape([-1])]
    nframes = len(positions)
    if nframes >=1 :
        data['nframes'] = nframes
        data['positions'] = np.vstack(positions)
        data['forces'] = np.vstack(forces)
        data['energies'] = np.hstack(energies)
        data['cells'] = np.vstack(cells)
    else:
        return {'nframes':nframes}


    return data

if __name__ == '__main__':
    main()
