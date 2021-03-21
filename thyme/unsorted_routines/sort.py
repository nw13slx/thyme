from glob import glob
from os import walk
from os.path import dirname, join, basename, isfile
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

plt.switch_backend("agg")

species = ["Ag"] * 143 + ["Pd"] * 7 + ["C"] * 3 + ["O"] * 3
natom = len(species)


def main():
    with open("npz/metadata.json") as fin:
        metadata = json.load(fin)
    alle = {}
    for folder in metadata["foldername"]:
        e_dict = pack_folder(folder, metadata)
        alle.update(e_dict)

    keys = np.sort(list(alle.keys()))
    list_pos = {}
    alldist = []
    ind = 0
    fout2 = open("sort_e.xyz", "w+")
    fout3 = open("all_sort_e.xyz", "w+")
    for k in keys:
        values = alle[k]
        pos0 = values["x0"]
        find = False
        ref = None
        xyz = values["x"].reshape([-1, 3])
        species = values["species"]
        natom = xyz.shape[0]
        print(natom, file=fout3)
        print(natom, file=fout3)
        for iatom in range(natom):
            print(
                species[iatom], xyz[iatom, 0], xyz[iatom, 1], xyz[iatom, 2], file=fout3
            )
        for kref in list_pos:
            pos = list_pos[kref]
            dist = np.linalg.norm(pos0[143:].reshape([-1]) - pos[143:].reshape([-1]))
            alldist += [dist]
            if dist < 10:
                find = True
                ref = kref
        if not find:
            with open(f"npz/sort_e{ind}.POSCAR", "w+") as fout:
                print(head, file=fout)
                list_pos[k] = pos0
                print("selected natom", natom)
                print("C", file=fout)
                for iatom in range(natom):
                    print(xyz[iatom, 0], xyz[iatom, 1], xyz[iatom, 2], file=fout)
                print(natom, file=fout2)
                print(natom, file=fout2)
                for iatom in range(natom):
                    print(
                        species[iatom],
                        xyz[iatom, 0],
                        xyz[iatom, 1],
                        xyz[iatom, 2],
                        file=fout2,
                    )
            ind += 1
        else:
            print("found before", k, ref)
    fout2.close()
    fout3.close()
    fig = plt.figure()
    plt.hist(alldist)
    fig.savefig("dist_hist.png", dpi=300)
    print(np.sort(alldist)[:100])


def pack_folder(folder, metadata):

    e_dict = {}
    casename = "_".join(folder.split("/"))

    if isfile(f"npz/{casename}.npz"):

        data = np.load(f"npz/{casename}.npz", allow_pickle=True)
        energy = data["pe"]
        forces = data["forces"]
        positions = data["positions"]
        cell = data["cell"]
        species = data["species"]
        meta = data["metadata"]
        index = np.argmin(energy)
        e_dict[energy[index]] = {
            "x": positions[index],
            "x0": data["original_slab"],
            "f": forces[index],
            "cell": cell,
            "species": species,
            "metadata": meta,
        }
    else:
        print(folder, f"npz/{casename}.npz")
    return e_dict


head = """
1
17.627098    0.000000    0.000000
8.852748   15.333410    0.000000
0.000000    0.000000   25.584015
Ag   Pd   C   O
143   7   3   3  """

if __name__ == "__main__":
    main()
