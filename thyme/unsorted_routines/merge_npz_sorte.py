import json
import logging
import numpy as np
import re
import sys

from collections import Counter
from glob import glob
from os import walk, mkdir
from shutil import move
from os.path import dirname, join, basename, isdir, isfile

rootfolder = sys.argv[1]
if not isdir(rootfolder):
    mkdir(rootfolder)
logging.basicConfig(
    filename=f"{rootfolder}/merge.log", filemode="w", level=logging.INFO
)


def main():

    npzs = glob(f"*/allmerged.npz")

    fout = open(rootfolder + "/everything.xyz", "w+")

    max_atom = 0
    for npz in npzs:
        data = np.load(npz)
        natom = data["positions"].shape[1] // 3
        if natom > max_atom:
            max_atom = natom

    positions = {}
    forces = {}
    cells = {}
    symbols = {}
    atomic_number = {}
    total_energy = {}
    ele_to_N = {"Ag": 46, "Pd": 47, "C": 6, "O": 8}
    natoms = {}
    symbols = {}
    for npz in npzs:

        folder = dirname(npz)
        with open(f"{folder}/metadata.json") as fin:
            metadata = json.load(fin)

        data = np.load(npz)
        npzname = "_".join(npz.split("/"))
        # move(npz, f"{rootfolder}/{npzname}")

        o_pos = data["positions"]
        o_for = data["forces"]
        o_e = data["total_energy"]
        o_c = data["cells"]
        o_spe = metadata["species"]
        o_atomic = []
        for ele in o_spe:
            o_atomic += [ele_to_N[ele]]
        logging.info(f"{npz}, {natom}")

        label, legend = obtain_specie_label(metadata["species"])

        natom = data["positions"].shape[1] // 3
        nframes = data["positions"].shape[0]
        if natom < max_atom:
            o_pos = np.hstack([o_pos, np.zeros([nframes, (max_atom - natom) * 3])])
            o_for = np.hstack([o_for, np.zeros([nframes, (max_atom - natom) * 3])])
            o_atomic = np.hstack([o_atomic, [0] * (max_atom - natom)])
            o_spe = np.hstack([o_spe, ["NA"] * (max_atom - natom)])

        for n in range(nframes):
            print(max_atom, file=fout)
            print(o_e[n], file=fout)
            xyz = o_pos[n].reshape([-1, 3])
            for idx in range(max_atom):
                print(o_spe[idx], xyz[idx, 0], xyz[idx, 1], xyz[idx, 2], file=fout)
        if label not in positions:
            positions[label] = []
            forces[label] = []
            cells[label] = []
            total_energy[label] = []
            atomic_number[label] = []
            symbols[label] = []
            natoms[label] = []

        positions[label] += [o_pos]
        forces[label] += [o_for]
        cells[label] += [o_c]
        total_energy[label] += [o_e]
        atomic_number[label] += [np.hstack([o_atomic] * nframes).reshape([nframes, -1])]
        symbols[label] += [np.hstack([o_spe] * nframes).reshape([nframes, -1])]
        natoms[label] += [[natom] * nframes]

    all_data = {}
    for task in ["lowe", "alle"]:
        all_data[task] = {}
        for dataset in ["train", "valid", "test"]:
            all_data[task][dataset] = {}
            for matrix in [
                "positions",
                "forces",
                "total_energy",
                "natoms",
                "cells",
                "symbols",
                "atomic_number",
            ]:
                all_data[task][dataset][matrix] = []

    for label in positions:

        positions[label] = np.vstack(positions[label])
        forces[label] = np.vstack(forces[label])
        total_energy[label] = np.hstack(total_energy[label])
        natoms[label] = np.hstack(natoms[label])
        cells[label] = np.vstack(cells[label])
        symbols[label] = np.vstack(symbols[label])
        atomic_number[label] = np.vstack(atomic_number[label])

        sort_id = np.argsort(total_energy[label])

        task = "lowe"
        for matrix in [
            "positions",
            "forces",
            "cells",
            "symbols",
            "atomic_number",
            "total_energy",
            "natoms",
        ]:
            d = locals()[matrix][label]
            for i, dataset in enumerate(["train", "valid", "test"]):
                all_data[task][dataset][matrix] += [d[sort_id[i:42:3]]]

        stride = len(sort_id) // 42
        task = "alle"
        for matrix in [
            "positions",
            "forces",
            "cells",
            "symbols",
            "atomic_number",
            "total_energy",
            "natoms",
        ]:
            d = locals()[matrix][label]
            for i, dataset in enumerate(["train", "valid", "test"]):
                all_data[task][dataset][matrix] += [d[sort_id[i :: stride * 3]]]

        logging.info(
            f"{label}: {positions[label].shape}, {total_energy[label][sort_id[0]]}"
            f"{total_energy[label][sort_id[-1]]}"
        )

    for task in ["lowe", "alle"]:
        for dataset in ["train", "valid", "test"]:
            for matrix in ["positions", "forces", "cells", "symbols", "atomic_number"]:
                all_data[task][dataset][matrix] = np.vstack(
                    all_data[task][dataset][matrix]
                )
            for matrix in ["total_energy", "natoms"]:
                all_data[task][dataset][matrix] = np.hstack(
                    all_data[task][dataset][matrix]
                )
            logging.info(
                f"{task} {dataset} {all_data[task][dataset]['positions'].shape}"
            )
        save_data = {}
        for matrix in ["positions", "forces", "cells", "symbols", "atomic_number"]:
            save_data[matrix] = np.vstack(
                [
                    all_data[task]["train"][matrix],
                    all_data[task]["valid"][matrix],
                    all_data[task]["test"][matrix],
                ]
            )
        for matrix in ["total_energy", "natoms"]:
            save_data[matrix] = np.hstack(
                [
                    all_data[task]["train"][matrix],
                    all_data[task]["valid"][matrix],
                    all_data[task]["test"][matrix],
                ]
            )

        np.savez(f"{rootfolder}/{task}.npz", **save_data)
    fout.close()


def obtain_specie_label(species):
    index_list = {}
    ele_list = {}
    for ele in set(species):
        index_list[ele] = np.array(
            [index for index, e in enumerate(species) if e == ele]
        )
        ele_list[ele] = len(index_list[ele])
    sort_ele = np.array(sorted(ele_list.items(), key=lambda x: x[0]))
    label = ""
    legend = ""
    for ele in sort_ele:
        label += f"{ele[0]}{ele[1]}"
        legend += ele[0] + "$_{" + str(ele[1]) + "}$"
    return label, legend


if __name__ == "__main__":
    main()
