"""
Data utils.

Lixin Sun, MIR Group, Harvard U
"""
import logging
import numpy as np


def convert_cell_format(nframes, raw_cell):
    """
    Args:

    """

    input_cell = np.array(raw_cell, dtype=np.float32)
    n_dim = len(input_cell.shape)
    n_elements = (input_cell.reshape([-1])).shape[0]

    if n_elements == 1:

        cell = np.zeros((3, 3))
        cell[0, 0] = input_cell[0]
        cell[1, 1] = input_cell[0]
        cell[2, 2] = input_cell[0]

        if n_dim > 1:
            cell = cell.reshape((1,) + cell.shape)

    elif n_elements == 3:

        cell = np.zeros((3, 3))
        cell[0, 0] = input_cell[0]
        cell[1, 1] = input_cell[1]
        cell[2, 2] = input_cell[2]

        if n_dim > 1:
            cell = cell.reshape((1,) + cell.shape)

    elif n_elements == 9:

        cell = input_cell.reshape([3, 3])

        if n_dim > 2 or (n_dim == 2 and input_cell.shape != (3, 3)):
            cell = cell.reshape((1,) + cell.shape)

    elif n_elements == 3 * nframes:

        cell = np.zeros((nframes, 3, 3))
        for idx, cell in enumerate(input_cell):
            cell[idx, 0, 0] = cell[0]
            cell[idx, 1, 1] = cell[1]
            cell[idx, 2, 2] = cell[2]

    elif n_elements == 9 * nframes:

        cell = input_cell.reshape([nframes, 3, 3])

    elif n_elements == 6 * nframes:

        raise NotImplementedError(
            f"(abc alpha, beta, gamma) cell form is not implemented"
        )

    else:

        raise RuntimeError(f"the input cell shape {input_cell.shape} does not work")

    return cell
