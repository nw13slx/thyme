import json
import numpy as np
import re

from collections import Counter
from glob import glob
from os import walk, mkdir
from os.path import dirname, join, basename, isdir, isfile


def main():
    train_npzs = glob(f"*/train.npz")
    valid_npzs = glob(f"*/valid.npz")
    test_npzs = glob(f"*/test.npz")

    fout = open("everything.xyz", "w+")

    max_atom = 0
    for npz in train_npzs+test_npzs+valid_npzs:
        data = np.load(npz)
        natom = data['positions'].shape[1] // 3
        if natom > max_atom:
            max_atom = natom

    nframes = 0
    for npz in train_npzs:
        data = np.load(npz)
        nframes += data['positions'].shape[0]
    print("train", nframes)

    nframes = 0
    for npz in test_npzs:
        data = np.load(npz)
        nframes += data['positions'].shape[0]
    print("test", nframes)

    nframes = 0
    for npz in valid_npzs:
        data = np.load(npz)
        nframes += data['positions'].shape[0]
    print("valid", nframes)

    ele_to_N = {'Ag':46, 'Pd':47, 'C':6, 'O':8}
    for npz in train_npzs+valid_npzs+test_npzs:

        folder = dirname(npz)
        with open(f"{folder}/metadata.json") as fin:
            metadata = json.load(fin)

        data = np.load(npz)
        o_pos = data['positions']
        o_for = data['forces']
        o_e = data['energies']
        o_c = data['cells']
        o_spe = metadata['species']
        o_atomic = []
        for ele in o_spe:
            o_atomic += [ele_to_N[ele]]
        print(npz, natom)

        natom = data['positions'].shape[1] // 3
        nframes = data['positions'].shape[0]
        if natom < max_atom:
            o_pos = np.hstack([o_pos, np.zeros([nframes, (max_atom-natom)*3])])
            o_for = np.hstack([o_for, np.zeros([nframes, (max_atom-natom)*3])])
            o_atomic = np.hstack([o_atomic, [0]*(max_atom-natom)])
            o_spe = np.hstack([o_spe, ['NA']*(max_atom-natom)])

        for n in range(nframes):
            print(o_pos.shape, o_pos[n].shape, n, nframes, max_atom)
            print(max_atom, file=fout)
            print(o_e[n], file=fout)
            xyz = o_pos[n].reshape([-1, 3])
            for idx in range(max_atom):
                print(o_spe[idx], xyz[idx, 0], xyz[idx, 1],
                      xyz[idx, 2], file=fout)
            xyz = None
    fout.close()




if __name__ == '__main__':
    main()
