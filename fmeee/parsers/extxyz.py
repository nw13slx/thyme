import logging
import numpy as np

from fmeee.parsers.monty import read_pattern, read_table_pattern
from fmeee.trajectory import PaddedTrajectory

def extxyz_to_padded_trj(filename):
    logging.info(f"converting {filename}")
    fl_num = r"([+-]?\d+.\d+[eE]?[+-]?\d*)"
    sfl_num = r"\s+([+-]?\d+.\d+[eE]?[+-]?\d*)"
    d = \
        read_pattern(filename,
                     {'natoms':r"^([0-9]+)$",
                      'cells':r"Lattice=\""+fl_num+sfl_num+sfl_num \
                      +sfl_num+sfl_num+sfl_num \
                      +sfl_num+sfl_num+sfl_num+r"\"",
                      'energies':r"free_energy="+fl_num,
                      'posforce': r"^\w+"+sfl_num+sfl_num+sfl_num \
                      +"\s+[A-Z]\s+\d"+sfl_num+sfl_num+sfl_num,
                      'symbols':r"^([a-zA-Z]+)\s"
                      })
    natoms = np.array(d['natoms'], dtype=int).reshape([-1])
    cells = np.array(d['cells'], dtype=float).reshape([-1, 3, 3])
    energies = np.array(d['energies'], dtype=float).reshape([-1])
    posforce = np.array(d['posforce'], dtype=float).reshape([-1, 6])

    positions = posforce[:, :3]
    forces = posforce[:, 3:]
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

#  71
#   Lattice="9.18446134 0.0 0.0 0.0 22.63986204 -4.15391219 0.0 0.0 29.07738533"                  Properties=species:S:1:pos:R:3:move_mask:L:1:tags:I:1:forces:R:3 energy=-181.54722937         free_energy=-181.54878652 pbc="T T T"
#
