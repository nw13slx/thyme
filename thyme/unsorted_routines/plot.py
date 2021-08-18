from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ase.io.extxyz import write_xyz
from ase.atoms import Atoms
import logging

logging.basicConfig(
    filename=f"plot.log", filemode="w", level=logging.INFO, format="%(message)s"
)
logging.getLogger().addHandler(logging.StreamHandler())

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
    for npz in glob("all_*.npz"):
        data = np.load(npz, allow_pickle=True)
        label = npz[4:-4]
        fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
        axs[0].hist(data["total_energy"], log=True, bins=50)
        axs[1].hist(data["forces"].reshape([-1]), log=True, bins=50)
        axs[0].set_xlabel("Energy (eV)")
        axs[1].set_xlabel("Force (eV)")
        axs[0].set_ylabel("counts")
        axs[1].set_ylabel("counts")
        fig.tight_layout()
        fig.savefig(f"hist_{label}.png", dpi=300)
        del axs
        del fig
        plt.close()

        max_total_energy = np.argsort(data["total_energy"])
        # max_forces = np.max(np.abs(data['forces']), axis=-1)
        # max_forces = np.argsort(max_forces)
        # max_to_print = np.min([5, len(max_total_energy)])
        # idcs = set(list(np.hstack([max_forces[-max_to_print:], max_total_energy[-max_to_print:]])))

        positions = data["positions"]
        cells = data["cells"]
        count = 0
        for index in max_total_energy:
            atoms = Atoms(
                data["species"],
                positions=positions[index].reshape([-1, 3]),
                cell=cells[index].reshape([3, 3]),
                pbc=True,
                info={"energy": data["total_energy"][index]},
            )
            write_xyz(f"{label}.xyz", atoms, comment=data["names"][index], append=True)
            logging.info(
                f"{label} {count} {data['names'][index]} {data['total_energy'][index]}"
            )
            count += 1


if __name__ == "__main__":
    main()
