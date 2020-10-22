import numpy as np
from collections import Counter
from ase.data import atomic_numbers

atomic_numbers_dict = atomic_numbers
atomic_numbers_dict.update(
    {'0': 0,
     'NA':0,
     0: 0}
)

def species_to_order_label(symbol):

    count = dict(Counter(symbol))
    for i in ['0', 0, 'NA']:
        if i in count:
            del count[i]

    symbol_list = []
    number_list = []
    for k in count:
       symbol_list += [k]
       number_list += [atomic_numbers_dict[k]]
    symbol_list = [symbol_list[i] for i in np.argsort(number_list)]

    order = []
    for k in symbol_list:
        order += [[i for i, s in enumerate(symbol) if s==k]]
    order = np.hstack(order)
    label = "".join([f"{k}{count[k]}" for k in symbol_list])
    return order, label

def species_to_idgroups(symbol):

    idgroups = []
    symbol_list = list(set(list(symbol)))
    for k in symbol_list:
        idgroups += [[i for i, s in enumerate(symbol) if s==k]]
    return symbol_list, idgroups

def convert_species(nframes, raw_n_atoms, raw_data):
    """
    convert raw_species array into a (nframes, n_atoms) matrix
    """

    # assuming raw_data is a formated array
    raw_atoms_pattern = np.array(raw_data)
    if not isinstance(raw_atoms_pattern.reshape([-1])[0], int):
        raw_atomic_number = np.zeros_like(raw_atoms_pattern, dtype=np.int)
        for idx in itertools.product(*[range(s) for s in raw_atoms_pattern.shape]):
            raw_atomic_number[idx] = atomic_numbers_dict[raw_atoms_pattern[idx]]
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
        atoms_pattern = np.array([[first_number]*n_atoms]*nframes,
                                 dtype=np.int)
        list_of_symbols = np.array([[first_element]*n_atoms]*nframes,
                                   dtype=np.int)

    elif n_elements == n_atoms:
        atoms_pattern = np.array([raw_atomic_number.reshape([n_atoms])]*nframes,
                                 dtype=np.int)
        list_of_symbols = np.array([raw_atoms_pattern.reshape([n_atoms])]*nframes,
                                   dtype=np.int)

    elif n_elements == nframes*n_atoms:
        atoms_pattern = raw_atomic_number.reshape([nframes, n_atoms])
        list_of_symbols = raw_atoms_pattern.reshape([nframes, n_atoms])

    else:

        raise RuntimeError(f"the input cell shape {input_cell.shape} does not work")

    return atoms_pattern, list_of_symbols
