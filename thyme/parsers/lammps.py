import logging
import numpy as np

from glob import glob
from os.path import getmtime, isfile
from os import remove

from thyme.parsers.monty import read_pattern, read_table_pattern
from thyme.trajectory import PaddedTrajectory
from thyme.routines.folders import find_folders, find_folders_matching

fl_num = r"([+-]?\d+.\d+[eE]?[+-]?\d*)"
sfl_num = r"\s+([+-]?\d+.\d+[eE]?[+-]?\d*)"
nc_fl_num = r"[+-]?\d+.\d+[eE]?[+-]?\d*"
head_str = """ITEM: TIMESTEP
{timestep}
ITEM: NUMBER OF ATOMS
{natom}
ITEM: BOX BOUNDS pp pp pp
0 {lx}
0 {ly}
0 {lz}
ITEM: ATOMS id x y z {type_str}"""


def write(name, trj, color_key=""):
    if isfile(name):
        remove(name)

    for key in trj.per_frame_attrs:

        if key == "forces":
            type_str = "fx fy fz"
        elif key == "velocities":
            type_str = "vx vy vz"
        elif key == color_key:
            type_str = "q"

    if not trj.is_padded:
        species = np.unique(trj.symbolx)
    else:
        species = np.unique(trj.species)

    fout = open(name, "w+")

    natom = trj.natom
    if not trj.is_padded:
        atomic_number = np.array(
            [atomic_numbers_dict[sym] for sym in trj.species], dtype=int
        ).reshape([-1, 1])

    for i in range(trj.nframes):

        non_diag = trj.cells[i][((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))]
        if np.max(np.abs(non_diag)) > 0:
            raise NotImplementedError()

        if trj.is_padded:
            natom = trj.natoms[i]
            atomic_number = np.array(
                [atomic_numbers_dict[sym] for sym in trj.symbols[i]], dtype=int
            ).reshape([-1, 1])

        head_str.format(
            dict(
                lx=trj.cells[i][0, 0],
                ly=trj.cells[i][1, 1],
                lz=trj.cells[i][2, 2],
                timestep=i,
                natom=natom,
                type_str=type_str,
            )
        )
        #  data
        # for key in trj.per_frame_attrs:

    logging.info(f"write {name}")
    fout.close()


def write_trjs(name, trjs):
    for i, trj in trjs.alldata.items():
        write(f"{trj.name}_{name}", trj)
