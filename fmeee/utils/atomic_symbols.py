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
    if 'NA' in count:
        del count['NA']

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
    symbol_list = set(list(symbol))
    for k in symbol_list:
        idgroups += [[i for i, s in enumerate(symbol) if s==k]]
    return symbol_list, idgroups
