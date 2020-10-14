"""
Data utils.

Lixin Sun, MIR Group, Harvard U
"""
import logging
import numpy as np

def convert_cell_format(nframes, raw_cells):
    """
    Args:

    """

    input_cell = np.array(raw_cells, dtype=np.float32)
    n_elements = (input_cell.reshape([-1])).shape[0]

    if n_elements == 1:

        cell = np.zeros((3,3))
        cell[0, 0] = input_cell[0]
        cell[1, 1] = input_cell[0]
        cell[2, 2] = input_cell[0]

        cells = np.tile(np.expand_dims(cell, axis=0), (nframes, 1, 1))

    elif n_elements == 3:

        cell = np.zeros((3,3))
        cell[0, 0] = input_cell[0]
        cell[1, 1] = input_cell[1]
        cell[2, 2] = input_cell[2]

        cells = np.tile(np.expand_dims(cell, axis=0), (nframes, 1, 1))

    elif n_elements == 9:

        cells = np.tile(np.expand_dims(input_cell.reshape([3, 3]),
                                       axis=0),
                        (nframes, 1, 1))

    elif n_elements == 3*nframes:

        cells = np.zeros((nframes, 3, 3))
        for idx, cell in enumerate(input_cell):
            cells[idx, 0, 0] = cell[0]
            cells[idx, 1, 1] = cell[1]
            cells[idx, 2, 2] = cell[2]

    elif n_elements == 9 * nframes:

        cells = input_cell.reshape([nframes, 3, 3])

    elif n_elements == 6 * nframes:

        raise NotImplementedError(f"(abc alpha, beta, gamma) cell form is not implemented")

    else:

        raise RuntimeError(f"the input cell shape {input_cell.shape} does not work")

    return cells
