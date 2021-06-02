import logging
import numpy as np

from glob import glob
from os.path import getmtime, isfile
from os import remove

from ase.atoms import Atoms
from ase.io.extxyz import key_val_str_to_dict, parse_properties
from ase.io.extxyz import write_xyz as write_extxyz
from ase.calculators.singlepoint import SinglePointCalculator

from thyme.parsers.monty import read_pattern, read_table_pattern
from thyme.trajectory import PaddedTrajectory
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

    hasxyz = len(xyzs) > 0
    if not hasxyz:
        return data

    xyzs = sorted(xyzs, key=getmtime)

    join_trj = PaddedTrajectory()
    for filename in xyzs:
        join_trj.add_trj(extxyz_to_padded_trj(filename, data_filter))

    return join_trj


def pack_folder(folder, data_filter=None, include_xyz=True):

    join_trj = pack_folder_trj(folder, data_filter, include_xyz)

    data = join_trj.to_dict()

    return data


def extxyz_to_padded_dict(filename):

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
            "free_energies": r"free_energy=" + fl_num,
            "energies": r"energy=" + fl_num,
            "posforce": string,
            "symbols": r"^([a-zA-Z]+)\s",
        },
    )

    natoms = np.array(d["natoms"], dtype=int).reshape([-1])
    # logging.debug(f"found {len(natoms)} frames with maximum {np.max(natoms)} atoms")

    if len(d["free_energies"]) > 0:
        energies = np.array(d["free_energies"], dtype=float).reshape([-1])
        logging.debug("use free_energies tag for energies")
    else:
        energies = np.array(d["energies"], dtype=float).reshape([-1])
        logging.debug("use energies tag for energies")

    cells = np.array(d["cells"], dtype=float).reshape([-1, 3, 3])

    posforce = np.array(d["posforce"], dtype=float).reshape([-1, 6])
    positions = posforce[:, index["pos"] : index["pos"] + 3]
    forces = posforce[:, index["forces"] : index["forces"] + 3]
    # logging.debug(f"pos.shape {positions.shape} force.shape {forces.shape}")
    # logging.debug(f"first couple lines of posforce")
    # logging.debug(f"{posforce[0]}")
    # logging.debug(f"{posforce[1]}")
    # logging.debug(f"{posforce[2]}")

    symbols = np.array(d["symbols"], dtype=str).reshape([-1])

    max_atoms = np.max(natoms)
    newpos = []
    newforce = []
    newsymbols = []
    counter = 0
    for i, natom in enumerate(natoms):
        pos = np.zeros((max_atoms, 3))
        pos[:natom] += positions[counter : counter + natom]
        newpos += [[pos]]
        fo = np.zeros((max_atoms, 3))
        fo[:natom] += forces[counter : counter + natom]
        newforce += [[fo]]
        newsymbols += [
            np.hstack((symbols[counter : counter + natom], ["0"] * (max_atoms - natom)))
        ]
        counter += natom
    positions = np.vstack(newpos)
    forces = np.vstack(newforce)
    symbols = np.vstack(newsymbols)

    # logging.debug(f"after reshape: pos.shape {positions.shape} force.shape {forces.shape}")
    # logging.debug(f"first couple lines of posforce")
    # logging.debug(f"{positions[0][0]}")
    # logging.debug(f"{positions[0][1]}")
    # logging.debug(f"{positions[0][2]}")

    dictionary = dict(
        positions=positions,
        forces=forces,
        energies=energies,
        cells=cells,
        symbols=symbols,
        natoms=natoms,
        natom=max_atoms,
    )

    # double check all arrays have the same number of frames
    nframes = []
    for k in dictionary:
        if k != "natom":
            nframes += [dictionary[k].shape[0]]
    assert len(set(nframes)) == 1

    return dictionary


def extxyz_to_padded_trj(filename, data_filter=None):

    dictionary = extxyz_to_padded_dict(filename)

    trj = PaddedTrajectory.from_dict(dictionary)

    if data_filter is not None:
        try:
            accept_id = data_filter(trj)
            trj.filter_frames(accept_id)
        except Exception as e:
            logging.error(f"{e}")
            logging.error("extxyz only accept batch filter work on paddedtrajectory")
            raise RuntimeError("")

    trj.name = filename

    logging.info(f"convert {filename} to {repr(trj)}")
    logging.debug(f"{trj}")

    return trj


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


def write(name, trj):
    if isfile(name):
        remove(name)
    if not trj.is_padded:
        for i in range(trj.nframes):
            definition = {'pbc':False}
            if 'cells' in trj.per_frame_attrs:
                definition['cell']=trj.cells[i]
                definition['pbc']=True
            structure = Atoms(
                symbols=trj.species,
                positions=trj.positions[i],
                **definition
            )
            definition = {'forces':trj.forces[i]} if 'forces' in trj.per_frame_attrs else {}
            calc = SinglePointCalculator(
                structure, energy=trj.energies[i], **definition
            )
            structure.calc = calc
            write_extxyz(name, structure, append=True)
    else:
        for i in range(trj.nframes):
            natom = trj.natoms[i]
            structure = Atoms(
                cell=trj.cells[i],
                symbols=trj.symbols[i][:natom],
                positions=trj.positions[i][:natom],
                pbc=True,
            )
            calc = SinglePointCalculator(
                structure, energy=trj.energies[i], forces=trj.forces[i][:natom]
            )
            structure.calc = calc
            write_extxyz(name, structure, append=True)
    logging.info(f"write {name}")


def write_trjs(name, trjs):
    for i, trj in trjs.alldata.items():
        write(f"{trj.name}_{name}", trj)
