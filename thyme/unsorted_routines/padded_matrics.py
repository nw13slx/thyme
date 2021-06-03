import time
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ase.io.extxyz import write_xyz
from ase.atoms import Atoms
import logging

logging.basicConfig(
    filename=f"padded.log", filemode="a", level=logging.INFO, format="%(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("----")

matplotlib.use("Agg")


matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["font.size"] = 8

# https://matplotlib.org/3.1.0/gallery/color/named_colors.html

tabcolors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


def main():

    logging.info(f"{time.time()}")
    with open("list") as fin:
        lines = fin.readlines()
    maxatom = 0
    for line in lines:
        label = line.split()[0][4:-4]
        s = ""
        for i in label:
            if i.isalpha():
                s += " "
            else:
                s += i
        s = np.sum([int(i) for i in s.split()])
        print(label, s)
        if s > maxatom:
            maxatom = s

    positions = []
    cells = []
    total_energy = []
    forces = []
    symbols = []
    natoms = []

    re_positions = []
    re_cells = []
    re_total_energy = []
    re_forces = []
    re_symbols = []
    re_natoms = []
    for line in lines:
        l = line.split()
        npz = l[0]
        data = np.load(npz, allow_pickle=True)
        label = npz[4:-4]

        max_total_energy = np.argsort(data["total_energy"])
        skip = float(l[1])
        remove_top = int(l[2])
        nframes = data["cells"].shape[0]

        if remove_top == 0:
            if skip >= 1:
                skip = int(skip)
                idlist = max_total_energy[::skip]
                if (nframes - 1) not in idlist:
                    idlist = np.hstack([idlist, [nframes - 1]])
                remain_list = set(np.arange(nframes)) - set(idlist)
            else:
                portion = int(np.floor(nframes * skip))
                idlist = np.random.permutation(nframes)
                remain_list = idlist[:portion]
                idlist = idlist[portion:]
        else:
            if skip >= 1:
                skip = int(skip)
                idlist = max_total_energy[:-remove_top:skip]
                remain_list = set(max_total_energy[:-remove_top]) - set(idlist)
            else:
                nframes0 = nframes - remove_top
                portion = int(np.floor(nframes0 * skip))
                logging.info(f"fractional skip, total {nframes0}, portion {portion}")
                idlist = max_total_energy[np.random.permutation(nframes0)]
                remain_list = idlist[portion:]
                idlist = idlist[:portion]

        natom = len(data["species"])
        if natom < maxatom:
            symbol = np.hstack([data["species"], ["NA"] * (maxatom - natom)])
        else:
            symbol = data["species"]

        count = 0
        for index in idlist:

            cells += [data["cells"][index]]

            pos = np.zeros((maxatom, 3))
            pos[:natom, :] += data["positions"][index].reshape([-1, 3])
            positions += [pos.reshape([-1])]

            fo = np.zeros((maxatom, 3))
            fo[:natom, :] += data["forces"][index].reshape([-1, 3])
            forces += [fo.reshape([-1])]

            total_energy += [data["total_energy"][index]]

            symbols += [symbol]
            natoms += [natom]
            count += 1

        ori_nframes = data["cells"].shape[0]
        logging.info(
            f"{label} {count} from {ori_nframes} skip {skip} remove_top {remove_top}"
        )

        count = 0
        for index in remain_list:

            re_cells += [data["cells"][index]]

            pos = np.zeros((maxatom, 3))
            pos[:natom, :] += data["positions"][index].reshape([-1, 3])
            re_positions += [pos.reshape([-1])]

            fo = np.zeros((maxatom, 3))
            fo[:natom, :] += data["forces"][index].reshape([-1, 3])
            re_forces += [fo.reshape([-1])]

            re_total_energy += [data["total_energy"][index]]

            re_symbols += [symbol]
            re_natoms += [natom]
            count += 1
        logging.info(
            f"remain {label} {count} from {ori_nframes} skip {skip} remove_top {remove_top}"
        )

    positions = np.vstack(positions)
    forces = np.vstack(forces)
    cells = np.vstack(cells)
    symbols = np.vstack(symbols)
    natoms = np.hstack(natoms)
    total_energy = np.hstack(total_energy)

    nframes = cells.shape[0]
    nframes0 = cells.shape[0]
    permute_id = np.random.permutation(nframes)
    positions = positions[permute_id]
    forces = forces[permute_id]
    cells = cells[permute_id]
    symbols = symbols[permute_id]
    natoms = natoms[permute_id]
    total_energy = total_energy[permute_id]

    re_positions = np.vstack(re_positions)
    re_forces = np.vstack(re_forces)
    re_cells = np.vstack(re_cells)
    re_symbols = np.vstack(re_symbols)
    re_natoms = np.hstack(re_natoms)
    re_total_energy = np.hstack(re_total_energy)

    nframes = re_cells.shape[0]
    permute_id = np.random.permutation(nframes)
    re_positions = re_positions[permute_id]
    re_forces = re_forces[permute_id]
    re_cells = re_cells[permute_id]
    re_symbols = re_symbols[permute_id]
    re_natoms = re_natoms[permute_id]
    re_total_energy = re_total_energy[permute_id]

    logging.info(
        f"save as {nframes0}_frames.npz with {nframes0} training and {nframes} validation"
    )
    np.savez(
        f"{nframes0}_frames.npz",
        positions=np.vstack([positions, re_positions]),
        forces=np.vstack([forces, re_forces]),
        cells=np.vstack([cells, re_cells]),
        symbols=np.vstack([symbols, re_symbols]),
        natoms=np.hstack([natoms, re_natoms]),
        total_energy=np.hstack([total_energy, re_total_energy]),
    )


if __name__ == "__main__":
    main()
