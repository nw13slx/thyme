import logging
import numpy as np

from glob import glob
from os.path import getmtime, isfile
from os import remove

from ase.atoms import Atoms
from ase.io.extxyz import key_val_str_to_dict, parse_properties
from ase.io.extxyz import write_xyz as write_extxyz
from ase.calculators.singlepoint import SinglePointCalculator

from thyme import Trajectories, Trajectory
from thyme._key import *
from thyme.parsers.monty import read_pattern, read_table_pattern
from thyme.routines.folders import find_folders, find_folders_matching

fl_num = r"([+-]?\d+.\d+[eE]?[+-]?\d*)"
sfl_num = r"\s+([+-]?\d+.\d+[eE]?[+-]?\d*)"
nc_fl_num = r"[+-]?\d+.\d+[eE]?[+-]?\d*"


def get_childfolders(path, include_xyz=True):

    if include_xyz:
        return find_folders_matching(["*.xyz", "*.extxyz"], path)
    else:
        return find_folders_matching(["*.extxyz"], path)


def pack_folder_trj(folder, data_filter=None, include_xyz=True):

    xyzs = glob(f"{folder}/*.extxyz")
    if include_xyz:
        xyzs += glob(f"{folder}/*.xyz")

    if len(xyzs) == 0:
        return Trajectories()

    xyzs = sorted(xyzs, key=getmtime)

    join_trj = Trajectories()
    for filename in xyzs:
        join_trj.add_trjs(extxyz_to_trjs(filename, data_filter))

    return join_trj


def pack_folder(folder, data_filter=None, include_xyz=True):

    join_trj = pack_folder_trj(folder, data_filter, include_xyz)

    data = join_trj.to_dict()

    return data


def extxyz_to_trjs(filename, data_filter=None):

    string, index = posforce_regex(filename)
    logging.debug(f"use regex {string} to parse for posforce")
    logging.debug(f"posindex {index}")

    logging.info(f"converting {filename}")
    d = read_pattern(
        filename,
        {
            "natoms": r"^([0-9]+)$",
            "cells": r"Lattice=\""
            + fl_num
            + sfl_num
            + sfl_num
            + sfl_num
            + sfl_num
            + sfl_num
            + sfl_num
            + sfl_num
            + sfl_num
            + r"\"",
            "free_total_energy": r"free_energy=" + fl_num,
            "total_energy": r"energy=" + fl_num,
            "posforce": string,
            "symbols": r"^([a-zA-Z]+)\s",
        },
    )

    natoms = np.array(d["natoms"], dtype=int).reshape([-1])
    # logging.debug(f"found {len(natoms)} frames with maximum {np.max(natoms)} atoms")

    if len(d["free_total_energy"]) > 0:
        total_energy = np.array(d["free_total_energy"], dtype=float).reshape([-1])
        logging.debug("use free_total_energy tag for total_energy")
    else:
        total_energy = np.array(d["total_energy"], dtype=float).reshape([-1])
        logging.debug("use total_energy tag for total_energy")

    cell = np.array(d["cells"], dtype=float).reshape([-1, 3, 3])

    posforce = np.array(d["posforce"], dtype=float).reshape([-1, 6])
    position = posforce[:, index["pos"] : index["pos"] + 3]
    force = posforce[:, index["forces"] : index["forces"] + 3]
    del posforce

    species = np.array(d["symbols"], dtype=str).reshape([-1])

    max_atoms = np.max(natoms)
    newpos = []
    newforce = []
    newsymbols = []
    counter = 0
    trjs = Trajectories()
    for i, natom in enumerate(natoms):
        counter += natom
        trj = Trajectory.from_dict(
            {
                POSITION: position[counter : counter + natom].reshape([1, natom, 3]),
                FORCE: force[counter : counter + natom].reshape([1, natom, 3]),
                TOTAL_ENERGY: total_energy[[i]],
                CELL: cell[[i]],
                SPECIES: species[counter : counter + natom].reshape([1, natom]),
                PER_FRAME_ATTRS: [POSITION, FORCE, TOTAL_ENERGY, CELL, SPECIES],
            }
        )
        if data_filter is not None:
            try:
                accept_id = data_filter(trj)
                trj.include_frames(accept_id)
            except Exception as e:
                logging.error(f"{e}")
                raise RuntimeError(f"{e}")
        if trj.nframes > 0:
            trj.name = i
            trjs.add_trj(trj, merge=True, preserve_order=False)

    logging.info(f"convert {filename} to {repr(trjs)}")
    logging.debug(f"{trjs}")

    return trjs


def posforce_regex(filename):

    with open(filename) as fin:
        fin.readline()
        line = fin.readline()
        info = key_val_str_to_dict(line)
    properties, properties_list, dtype, convertesr = parse_properties(
        info["Properties"]
    )

    string = ""
    pos_id = -1
    forces_id = -1
    index = {"pos": 0, "forces": 0}
    item_count = 0
    for k, v in properties.items():
        length = v[1]
        if len(string) > 0:
            string += r"\s+"
        if k in ["pos", "forces"]:
            string += fl_num + r"\s+" + fl_num + r"\s+" + fl_num
            index[k] = item_count
        else:
            for i in range(length):
                if i > 0:
                    string += r"\s+"
                if convertesr[item_count + i] == str:
                    string += r"\w+"
                elif convertesr[item_count + i] == float:
                    string += nc_fl_num
                else:
                    logging.info(
                        f"parser is not implemented for type {convertesr[item_count+i]}"
                    )
        item_count += length
    if index["pos"] > index["forces"]:
        index["pos"] = 3
        index["forces"] = 0
    else:
        index["pos"] = 0
        index["forces"] = 3
    return string, index


def write(name, trj, append=False):
    if isfile(name) and not append:
        remove(name)
    for i in range(trj.nframes):
        definition = {"pbc": False}
        if CELL in trj.per_frame_attrs:
            definition["cell"] = trj.cell[i]
            definition["pbc"] = True
        structure = Atoms(symbols=trj.species, positions=trj.position[i], **definition)
        definition = {"forces": trj.force[i]} if FORCE in trj.per_frame_attrs else {}
        calc = SinglePointCalculator(
            structure, energy=trj.total_energy[i], **definition
        )
        structure.calc = calc
        write_extxyz(name, structure, append=True)
    logging.info(f"write {name}")


def write_trjs(name, trjs, joint=True):
    if isfile(name) and joint:
        remove(name)
    for i, trj in trjs.alltrjs.items():
        if joint:
            write(name, trj, append=True)
        else:
            write(f"{trj.name}_{name}", trj)
