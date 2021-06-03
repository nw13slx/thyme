import json
import numpy as np

from collections import Counter
from glob import glob
from os import walk, mkdir
from os.path import dirname, join, basename, isdir, isfile

from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar


def main():

    alldata = dict(np.load("alldata.npz", allow_pickle=True))

    trjnames = sort_filenames(alldata)

    positions = []
    forces = []
    total_energy = []
    cells = []
    ntrjs = 0
    merge_data = {}
    for trjname in trjnames:
        data = alldata[trjname].item()
        count = dict(Counter(data["species"]))
        label = "".join([f"{k}{count[k]}" for k in np.sort(list(count.keys()))])
        sort_id = np.argsort(data["species"])
        if label not in merge_data:
            merge_data[label] = {}
            for k in ["positions", "forces", "total_energy", "cells", "history"]:
                merge_data[label][k] = []
            merge_data[label]["species"] = np.sort(data["species"])

        nframes = data["positions"].shape[0]
        if nframes > 0:
            for k in ["positions", "forces"]:
                merge_data[label][k] += [
                    (data[k].reshape([nframes, -1, 3])[:, sort_id, :]).reshape(
                        [nframes, -1]
                    )
                ]
            for k in ["total_energy", "cells"]:
                merge_data[label][k] += [data[k]]
            names = [f"{trjname}_{i}" for i in range(nframes)]
            merge_data[label]["history"] += names
        ntrjs += 1

    for label in merge_data:
        for k in ["positions", "forces", "cells"]:
            merge_data[label][k] = np.vstack(merge_data[label][k])
        for k in ["total_energy"]:
            merge_data[label][k] = np.hstack(merge_data[label][k])
        np.savez(
            f"all_{label}.npz",
            species=merge_data[label]["species"],
            positions=merge_data[label]["positions"],
            forces=merge_data[label]["forces"],
            total_energy=merge_data[label]["total_energy"],
            cells=merge_data[label]["cells"],
            names=merge_data[label]["history"],
        )
        print(
            label, len(merge_data[label]["total_energy"]), len(merge_data[label]["history"])
        )

    return 0


def sort_filenames(data):

    filenames = []
    file_indices = []
    for trjname in data:
        try:
            trj_index = int(re.sub("[^\d]", "", trjname))
        except Exception as e:
            # print("failed", e)
            trj_index = -1
        filenames += [trjname]
        file_indices += [trj_index]
    filenames = np.array(filenames)
    file_indices = np.argsort(np.array(file_indices))
    return filenames[file_indices]


def filter_trj(data, data_filter):

    xyzs = data["positions"]
    fs = data["forces"]
    es = data["total_energy"]
    cs = data["cells"]
    nframes = xyzs.shape[0]

    positions = []
    forces = []
    total_energy = []
    cells = []
    for istep in range(nframes):
        xyz = xyzs[istep]
        f = fs[istep]
        e = es[istep]
        c = cs[istep]
        if data_filter(xyz, f, e, c, species):
            positions += [xyz.reshape([-1])]
            forces += [np.hstack(f)]
            total_energy += [e]
            cells += [c.reshape([-1])]
    nframes = len(positions)
    if nframes >= 1:
        data["positions"] = np.vstack(positions)
        data["forces"] = np.vstack(forces)
        data["total_energy"] = np.hstack(total_energy)
        data["cells"] = np.vstack(cells)
    else:
        return {}

    data["nframes"] = nframes

    return data


if __name__ == "__main__":
    main()
