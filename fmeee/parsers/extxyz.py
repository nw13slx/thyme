import logging
import numpy as np

from fmeee.parsers.monty import read_pattern, read_table_pattern
from fmeee.trajectory import PaddedTrajectory

from ase.io.extxyz import key_val_str_to_dict, parse_properties

fl_num = r"([+-]?\d+.\d+[eE]?[+-]?\d*)"
sfl_num = r"\s+([+-]?\d+.\d+[eE]?[+-]?\d*)"

def extxyz_to_padded_trj(filename):

    string, index = posforce_regex(filename)
    logging.debug(f"use regex {string} to parse for posforce")
    logging.debug(f"posindex {index}")

    logging.info(f"converting {filename}")
    d = \
        read_pattern(filename,
                     {'natoms':r"^([0-9]+)$",
                      'cells':r"Lattice=\""+fl_num+sfl_num+sfl_num \
                      +sfl_num+sfl_num+sfl_num \
                      +sfl_num+sfl_num+sfl_num+r"\"",
                      'free_energies':r"free_energy="+fl_num,
                      'energies':r"energy="+fl_num,
                      'posforce': string,
                      'symbols':r"^([a-zA-Z]+)\s"
                      })

    natoms = np.array(d['natoms'], dtype=int).reshape([-1])
    logging.debug(f"found {len(natoms)} frames with maximum {np.max(natoms)} atoms")

    if len(d['free_energies']) > 0:
        energies = np.array(d['free_energies'], dtype=float).reshape([-1])
        logging.debug("use free_energies tag for energies")
    else:
        energies = np.array(d['energies'], dtype=float).reshape([-1])
        logging.debug("use energies tag for energies")

    cells = np.array(d['cells'], dtype=float).reshape([-1, 3, 3])

    posforce = np.array(d['posforce'], dtype=float).reshape([-1, 6])
    positions = posforce[:, index['pos']:index['pos']+3]
    forces = posforce[:, index['forces']:index['forces']+3]
    logging.debug(f"pos.shape {positions.shape} force.shape {forces.shape}")

    symbols = np.array(d['symbols'], dtype=str).reshape([-1])

    max_atoms = np.max(natoms)
    newpos = []
    newforce = []
    newsymbols = []
    counter = 0
    for i, natom in enumerate(natoms):
        pos = np.zeros((max_atoms, 3))
        pos[:natom] += positions[counter:counter+natom]
        newpos += [[pos]]
        fo = np.zeros((max_atoms, 3))
        fo[:natom] += forces[counter:counter+natom]
        newforce += [[fo]]
        newsymbols += [np.hstack((symbols[counter:counter+natom],['0']*(max_atoms-natom)))]
        counter += natom
    positions = np.vstack(newpos)
    forces = np.vstack(newforce)
    symbols = np.vstack(newsymbols)


    dictionary = dict(
        positions = positions,
        forces = forces,
        energies = energies,
        cells=cells,
        symbols=symbols,
        natoms=natoms,
        natom=max_atoms
    )
    nframes = []
    for k in dictionary:
        if k!= 'natom':
            nframes += [dictionary[k].shape[0]]

    assert len(set(nframes)) == 1

    trj = PaddedTrajectory.from_dict(dictionary)
    trj.name = filename

    logging.info(f"convert {filename} to {repr(trj)}")
    logging.debug(f"{trj}")

    return trj

def posforce_regex(filename):
    with open(filename) as fin:
        fin.readline()
        line = fin.readline()
        info = key_val_str_to_dict(line)
    properties, properties_list, dtype, convertesr \
        = parse_properties(info['Properties'])

    string = ""
    pos_id = -1
    forces_id = -1
    index = {'pos':0, 'forces':0}
    item_count = 0
    for k, v in properties.items():
        length = v[1]
        if len(string) > 0:
            string += r"\s+"
        if k in ['pos', 'forces']:
            string += fl_num+r"\s+"+fl_num+r"\s+"+fl_num
            index[k] = item_count
        else:
            for i in range(length):
                if i>0:
                    string += r'\s+'
                if convertesr[item_count+i] == str:
                    string += r'\w+'
                elif convertesr[item_count+i] == float:
                    string += fl_num
                else:
                    logging.info(f"parser is not implemented for type {convertesr[item_count+i]}")
        item_count += length
    if index['pos'] > index['forces']:
        index['pos'] = 3
        index['forces'] = 0
    else:
        index['pos'] = 0
        index['forces'] = 3
    return string, index
