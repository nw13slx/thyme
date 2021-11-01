import logging
import numpy as np

from glob import glob
from os.path import getmtime, isfile
from os import remove

from thyme.parsers.monty import read_pattern, read_table_pattern
from thyme.routines.folders import find_folders, find_folders_matching
from thyme._key import *

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
ITEM: ATOMS id type x y z {type_str}"""


def write(name, trj, color_key="", spe2num={}):

    if isfile(name):
        remove(name)

    keys = [POSITION]

    type_str = ""
    for key in trj.per_frame_attrs:

        if key == FORCE:
            type_str += " fx fy fz"
            keys += [FORCE]
        elif key == VELOCITY:
            type_str += " vx vy vz"
            keys += [VELOCITY]
        elif key == color_key:
            type_str += " q"
            keys += [color_key]

    fout = open(name, "w+")

    print("hello", trj.nframes)

    for i in range(trj.nframes):
        frame = trj.get_frame(i)

        cell = frame[CELL]
        off_dia_sum = np.sum(np.abs(cell)) - np.trace(np.abs(cell))
        if off_dia_sum > 0:
            raise NotImplementedError()

        natom = frame[NATOMS]
        hs = head_str.format(
            lx=cell[0, 0],
            ly=cell[1, 1],
            lz=cell[2, 2],
            timestep=i,
            natom=natom,
            type_str=type_str,
        )

        species = np.unique(frame[SPECIES])
        base = len(spe2num)
        if base == 0:
            base = 1
        spe2num.update(
            {spe: i + base for i, spe in enumerate(species) if spe not in spe2num}
        )

        string = f"{hs}"
        for j in range(natom):
            string += f"\n{j+1} {spe2num[frame[SPECIES][j]]} "
            for key in keys:
                string += " "+" ".join([f"{value}" for value in frame[key][j]])
        print(string, file=fout)

    logging.info(f"write {name}")
    fout.close()
    logging.info(f"spe2num {spe2num}")
