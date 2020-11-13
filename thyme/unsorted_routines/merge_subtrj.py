import json
import numpy as np

from collections import Counter
from glob import glob
from os import walk, mkdir
from os.path import dirname, join, basename, isdir, isfile

from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar


def o_dist_filter(xyz, f, e, c, species):
    C_id = np.array([index for index, ele in enumerate(species) if ele == 'C'])
    for Cindex in C_id:
        if xyz[Cindex, 2] < 10:
            print("skip frame for low C", flush=True)
            return False
    if (e+411 < -100) or (e+411 > 40):
        print("skip frame for high/low e", e, flush=True)
        return False
    return True


def main():

    alldata = dict(np.load("alldata.npz", allow_pickle=True))

    trjnames = sort_filenames(alldata)

    positions = []
    forces = []
    energies = []
    cells = []
    nframes = 0
    ntrjs = 0
    metadata = {}
    for trjname in trjnames:
        data = alldata[trjname].item()
        # alldata['trjname'] = filter_trj(data, data_filter)
        positions += [data['positions']]
        forces += [data['forces']]
        energies += [data['energies']]
        cells += [data['cells']]
        nframes += data['positions'].shape[0]
        # print(trjname, nframes)
        ntrjs += 1
        metadata.update(data)
    positions = np.vstack(positions)
    forces = np.vstack(forces)
    energies = np.hstack(energies)
    cells = np.vstack(cells)
    # print(positions.shape, forces.shape,
    #       energies.shape, cells.shape)
    np.savez("allmerged.npz", positions=positions,
             forces=forces, energies=energies,
             cells=cells)
    del metadata['positions']
    del metadata['forces']
    del metadata['energies']
    del metadata['cells']
    metadata['ntrjs'] = ntrjs
    metadata['nframes'] = nframes
    metadata['natoms'] = positions.shape[1]//3
    with open("metadata.json", "w+") as fout:
        json.dump(metadata, fout, indent=4)
    return 0


def sort_filenames(data):

    filenames = []
    file_indices = []
    for trjname in data:
        try:
            trj_index = int(re.sub('[^\d]', '', trjname))
        except Exception as e:
            # print("failed", e)
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
    if nframes >= 1:
        data['positions'] = np.vstack(positions)
        data['forces'] = np.vstack(forces)
        data['energies'] = np.hstack(energies)
        data['cells'] = np.vstack(cells)
    else:
        return {}

    data['nframes'] = nframes

    return data


if __name__ == '__main__':
    main()
