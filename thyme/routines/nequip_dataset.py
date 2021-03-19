"""
Dataset classes that parse array of positions, cells to AtomicData object
"""

import numpy as np

from ase.data import atomic_numbers

from nequip.data import NpzDataset, AtomicDataDict

atomic_numbers_dict = atomic_numbers
atomic_numbers_dict.update({"0": 0, "NA": 0, 0: 0})


class PaddedNpzDataset(NpzDataset):
    """Load data from an npz file.

    Args:
        file_name (str): file name of the npz file
        key_mapping (Dict[str, str]): mapping of npz keys to ``AtomicData`` keys
        force_fixed_keys (list): keys in the npz to treat as fixed quantities that don't change across examples. For example: cell, atomic_numbers
    """

    def get_data(self):

        fields, fixed_fields = NpzDataset.get_data(self)

        assert "natom" in fixed_fields
        assert "natoms" in fields, "PaddedNpzDataset needs natoms entry"

        natom = fixed_fields.pop("natom")
        natoms = fields.pop("natoms")
        nframes = fixed_fields.pop("nframes")

        for key, value in fields.items():
            if key != "natom" and len(value.shape) > 1:
                if value.shape[1] == natom:
                    fields[key] = [value[i, : natoms[i]] for i in range(nframes)]
        symbols = fields.pop("symbols")
        fields[AtomicDataDict.ATOMIC_NUMBERS_KEY] = [
            np.array([atomic_numbers_dict[sym] for sym in symbols[i]])
            for i in range(nframes)
        ]

        return fields, fixed_fields
