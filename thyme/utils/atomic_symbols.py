import numpy as np
from collections import Counter
from ase.data import atomic_numbers
import logging

atomic_numbers_dict = atomic_numbers
atomic_numbers_dict.update({"0": 0, "NA": 0, 0: 0})


def species_to_order_label(trj_species):

    counter = species_to_dict(trj_species)

    # sort atoms based on atomic number
    order = []
    for k in counter:
        order += [[i for i, s in enumerate(trj_species) if s == k]]
    order = np.hstack(order)
    label = "".join([f"{k}{counter[k]}" for k in counter])

    return order, label


def species_to_dict(trj_species):
    """
    given array of elements, return a dict with number for each species and sort the keys by atomic numbers
    """

    counter = dict(Counter(trj_species))
    for i in ["0", 0, "NA"]:
        if i in counter:
            del counter[i]

    # sort the formula by atomic number
    symbols = []
    atomic_numbers = []
    default_atomic_number = len(atomic_numbers_dict)
    for k in counter:
        symbols += [k]
        atomic_numbers += [atomic_numbers_dict.get(k, default_atomic_number)]
    symbols = [symbols[i] for i in np.argsort(atomic_numbers)]

    # obtain the new atomic order
    order_dict = {}
    for k in symbols:
        order_dict[k] = counter[k]

    return order_dict


def species_to_idgroups(symbol):
    """
    group atoms by elements, ordered by atomic numbers
    """

    idgroups = []
    symbols = species_to_dict
    for k in symbols:
        idgroups += [[i for i, s in enumerate(symbol) if s == k]]
    return list(symbols.keys()), idgroups


def convert_species(nframes, raw_n_atoms, raw_data):
    """
    convert raw_species array into a (nframes, n_atoms) matrix of atomic numbers
    """

    # assuming raw_data is a formated array
    raw_atoms_pattern = np.array(raw_data)
    default_atomic_number = len(atomic_numbers_dict)
    if not isinstance(raw_atoms_pattern.reshape([-1])[0], int):
        raw_atomic_number = np.zeros_like(raw_atoms_pattern, dtype=np.int)
        for idx in itertools.product(*[range(s) for s in raw_atoms_pattern.shape]):
            raw_atomic_number[idx] = atomic_numbers_dict.get(raw_atoms_pattern[idx], default_atomic_number)
    else:
        raw_atomic_number = raw_atoms_pattern

    n_elements = (raw_atoms_pattern.reshape([-1])).shape[0]

    if isinstance(raw_n_atoms, int):
        n_atoms = raw_n_atoms
    else:
        n_atoms = np.max(raw_n_atoms)

    if n_elements == 1:
        first_element = raw_atoms_pattern.reshape([-1])[0]
        first_number = raw_atomic_number.reshape([-1])[0]
        atoms_pattern = np.array([[first_number] * n_atoms] * nframes, dtype=np.int)
        symbols = np.array([[first_element] * n_atoms] * nframes, dtype=np.int)

    elif n_elements == n_atoms:
        atoms_pattern = np.array(
            [raw_atomic_number.reshape([n_atoms])] * nframes, dtype=np.int
        )
        symbols = np.array(
            [raw_atoms_pattern.reshape([n_atoms])] * nframes, dtype=np.int
        )

    elif n_elements == nframes * n_atoms:
        atoms_pattern = raw_atomic_number.reshape([nframes, n_atoms])
        symbols = raw_atoms_pattern.reshape([nframes, n_atoms])

    else:

        raise RuntimeError(f"the input cell shape {input_cell.shape} does not work")

    return atoms_pattern, symbols
