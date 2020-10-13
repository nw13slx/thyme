import logging
logging.basicConfig(filename=f'plot.log', filemode='w',
                    level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

from ase.atoms import Atoms
from ase.io.extxyz import write_xyz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from glob import glob


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 8

# https://matplotlib.org/3.1.0/gallery/color/named_colors.html

tabcolors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

def main():
    for npz in glob("all_*.npz"):
        data = np.load(npz, allow_pickle=True)
        label = npz[4:-4]
        fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
        axs[0].hist(data['energies'], log=True, bins=50)
        axs[1].hist(data['forces'].reshape([-1]), log=True, bins=50)
        axs[0].set_xlabel("Energy (eV)")
        axs[1].set_xlabel("Force (eV)")
        axs[0].set_ylabel("counts")
        axs[1].set_ylabel("counts")
        fig.tight_layout()
        fig.savefig(f"hist_{label}.png", dpi=300)
        del axs
        del fig
        plt.close()

        max_energies = np.argsort(data['energies'])
        # max_forces = np.max(np.abs(data['forces']), axis=-1)
        # max_forces = np.argsort(max_forces)
        # max_to_print = np.min([5, len(max_energies)])
        # idcs = set(list(np.hstack([max_forces[-max_to_print:], max_energies[-max_to_print:]])))

        positions = data['positions']
        cells = data['cells']
        count = 0
        for index in max_energies:
            atoms = Atoms(data['species'], positions=positions[index].reshape([-1, 3]),
                          cell=cells[index].reshape([3, 3]), pbc=True,
                          info={'energy':data['energies'][index]})
            write_xyz(f"{label}.xyz", atoms, comment=data['names'][index],
                      append=True)
            logging.info(f"{label} {count} {data['names'][index]} {data['energies'][index]}")
            count += 1


if __name__ == "__main__":
    main()
