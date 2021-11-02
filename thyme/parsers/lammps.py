import logging
import numpy as np

from glob import glob
from os.path import getmtime, isfile
from os import remove

from thyme import Trajectory
from thyme.parsers.monty import read_pattern, read_table_pattern
from thyme.routines.folders import find_folders, find_folders_matching
from thyme._key import *
from thyme.parsers.lammps_pizza_log import log as lammps_log
from thyme.parsers.lammps_pizza_dump import *

fl_num = r"([+-]?\d+.\d+[eE]?[+-]?\d*)"
sfl_num = r"\s+([+-]?\d+.\d+[eE]?[+-]?\d*)"
snum = r"\s+([+-]?\d+)"
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

def from_file(filename):

    data = dump(filename)
    col_id = data.names["id"]
    col_type = data.names["type"]
    x_id = data.names["x"]
    y_id = data.names["y"]
    z_id = data.names["z"]
    if "fx" in data.names:
        fx_id = data.names["fx"]
        fy_id = data.names["fy"]
        fz_id = data.names["fz"]

    trj = Trajectory()
    for i in range(data.nsnaps):
        if i%1000 == 0:
            logging.info(f"{i} / {data.nsnaps}")
        snap = data.snaps[i]
        cols = np.vstack(snap.atoms)
        ids = np.argsort(cols[:, col_id])
        species = cols[:, col_type][ids]
        pos = np.hstack((cols[:, x_id].reshape([-1, 1]),
                         cols[:, y_id].reshape([-1, 1]),
                         cols[:, z_id].reshape([-1, 1])))
        lx = snap.xhi-snap.xlo
        ly = snap.yhi-snap.ylo
        lz = snap.zhi-snap.zlo
        d = { CELL: np.diag([lx, ly, lz]).reshape([1, 3, 3]),
                POSITION: pos[ids].reshape([1, -1, 3]),
                SPECIES: species,
                PER_FRAME_ATTRS: [POSITION, CELL],
                FIXED_ATTRS: [SPECIES, NATOMS],
                }
        if "fx" in data.names:
            force = np.hstack((cols[:, fx_id].reshape([-1, 1]),
                               cols[:, fy_id].reshape([-1, 1]),
                               cols[:, fz_id].reshape([-1, 1])))[ids]
            d.update({FORCE: force.reshape([1, -1, 3])})
            d[PER_FRAME_ATTRS] += [FORCE]
        _trj = Trajectory.from_dict(d)
        trj.add_trj(_trj)
    return trj

def read_log(filename):
    l = lammps_log(filename, 0)
    l.next()
    data = np.array(l.data)
    return l.names, data
