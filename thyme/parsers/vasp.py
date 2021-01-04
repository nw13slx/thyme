import logging

import numpy as np
import time

from thyme.parsers.monty import read_pattern, read_table_pattern
from thyme.routines.folders import find_folders
from thyme.trajectory import Trajectory

from collections import Counter
from os.path import isfile

from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar
from ase.io.vasp import write_vasp


def get_childfolders(path):

    return find_folders(['vasprun.xml', 'OUTCAR', 'vasp.out'], path)


def pack_folder_trj(folder, data_filter):

    if folder == "./":
        folder = "."

    trj = parse_outcar_trj(folder, data_filter)
    if trj.nframes < 1:
        trj = parse_vasprun_trj(folder, data_filter)

    return trj


def parse_outcar_trj(folder, data_filter):

    data = {}

    filename = "/".join([folder, "KPOINTS"])
    if isfile(filename):
        kpoints = Kpoints.from_file(filename)
        data['kpoints'] = kpoints.kpts[0]

    filename = "/".join([folder, "CONTCAR"])
    filename2 = "/".join([folder, "POSCAR"])
    species = None
    if isfile(filename):
        try:
            poscar = Poscar.from_file(filename)
            species = [str(s) for s in poscar.structure.species]
        except Exception as e:
            logging.info(f"error loading contcar {e}")
    # else:
    #     logging.info("cannot find CONTCAR", filename)

    if species is None:
        if isfile(filename2):
            logging.info(filename2)
            try:
                poscar = Poscar.from_file(filename2)
                species = [str(s) for s in poscar.structure.species]
                data['species'] = species
            except Exception as e:
                logging.info(f"error loading poscar {e}")
                return Trajectory()
        # else:
        #     logging.info("cannot find", filename2)

    if species is None:
        logging.info("cannot find species in either POSCAR or CONTCAR")
        return Trajectory()

    # start parsing outcar
    filename = "/".join([folder, "OUTCAR"])

    t = time.time()
    d_energies = \
        read_pattern(filename,
                     {'energies':
                         r"free  energy   TOTEN\s+=\s+([+-]?\d+.\d+)"},
                     postprocess=lambda x: float(x))
    if len(d_energies['energies']) == 0:
        return Trajectory()
    energies = np.hstack(d_energies['energies'])

    logging.debug(f" parsing {filename} for positions")
    pos_force = read_table_pattern(filename,
                                   header_pattern=r"\sPOSITION\s+TOTAL-FORCE \(eV/Angst\)\n\s-+",
                                   row_pattern=r"^\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)$",
                                   footer_pattern=r"\s--+",
                                   postprocess=lambda x: float(x),
                                   last_one_only=False

                                   )
    pos_force = np.array(pos_force, dtype=np.float64)

    logging.debug(f" parsing {filename} for cells")
    cells = read_table_pattern(filename,
                               header_pattern=r"\sdirect lattice vectors\s+reciprocal lattice vectors",
                               row_pattern=r"\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)"
                               "\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)",
                               footer_pattern=r"^$",
                               postprocess=lambda x: float(x),
                               last_one_only=False
                               )
    cells = np.array(cells, dtype=np.float64)

    d_n_e_iter = read_pattern(filename,
                              {'n_e_iter': r"Iteration\s+(\d+)\s?\(\s+\d+\)"},
                              postprocess=lambda x: int(x))
    n_electronic_steps = np.array(
        d_n_e_iter['n_e_iter'], dtype=int).reshape([-1])
    logging.info(f"Outcar grep time {time.time()-t}")
    logging.info("loaded outcar")

    t = time.time()
    try:
        incar = Incar.from_file(filename)
        data['nelm'] = incar['NELM']
        data['cutoff'] = incar['ENCUT']
        data['dipole_correction'] = bool(incar['LDIPOL'].split()[0])
        data.update(incar)
        logging.info(f"Incar grep time {time.time()-t}")
    except Exception as e:
        logging.info(f"fail to load incar {e}")
        return Trajectory()

    nelm = data['nelm']

    cs = cells[:, :, :3].reshape([-1, 3, 3])

    # remove unconverged ionic steps
    c = Counter(n_electronic_steps)
    n_electronic_steps = [c[k] for k in sorted(c.keys())]
    nframes = pos_force.shape[0]
    converged_steps = np.array([i for i, s in enumerate(n_electronic_steps)
                                if (s < nelm and i < nframes)])

    # log and return tempty trajectory if needed
    if len(converged_steps) == 0:
        return Trajectory()

    elif len(converged_steps) < nframes:
        logging.info("skip unconverged step {}".format(
            [i for i, s in enumerate(n_electronic_steps)
             if (s >= nelm or i >= nframes)]))

    natom = pos_force.shape[1]
    data['natom'] = natom

    data.update(dict(cells=cs[converged_steps],
                     positions=pos_force[converged_steps, :, :3],
                     forces=pos_force[converged_steps, :, 3:],
                     energies=energies[converged_steps],
                     species=species
                     ))
    trj = Trajectory.from_dict(data)

    try:
        accept_id = data_filter(trj)
        trj.filter_frames(accept_id)
    except Exception as e:
        logging.error(f"{e}")
        logging.error(
            "extxyz only accept batch filter work on paddedtrajectory")
        raise RuntimeError("")

    trj.name = filename

    logging.info(f"convert {filename} to {repr(trj)}")
    logging.debug(f"{trj}")

    return trj


def parse_vasprun_trj(folder, data_filter):

    data = {}
    filename = "/".join([folder, "vasprun.xml"])
    if not isfile(filename):
        return Trajectory()

    try:
        vasprun = Vasprun(filename, ionic_step_skip=0,
                          exception_on_bad_xml=False)
    except Exception as e:
        logging.info("fail to load vasprun {e}")
        return Trajectory()

    nelm = vasprun.incar['NELM']
    data['nelm'] = nelm
    data['cutoff'] = vasprun.incar['ENCUT']
    data['dipole_correction'] = vasprun.parameters['LDIPOL']
    species = vasprun.atomic_symbols
    data['species'] = species
    data['natom'] = len(vasprun.atomic_symbols)
    data['kpoints'] = vasprun.kpoints.kpts[0]

    positions = []
    forces = []
    energies = []
    cells = []
    electronic_steps = []
    for step in vasprun.ionic_steps:
        electronic_steps += [len(step['electronic_steps'])]
        positions += [step['structure'].cart_coords.reshape([-1])]
        forces += [np.hstack(step['forces'])]
        energies += [step['e_fr_energy']]
        cells += [step['structure'].lattice.matrix.reshape([-1])]

    data.update(dict(cells=np.vstack(cells),
                     positions=np.vstack(positions),
                     forces=np.vstack(forces),
                     energies=np.hstack(energies),
                     species=species
                     ))
    trj = Trajectory.from_dict(data)

    # print(electronic_steps)
    # print(energies)
    electronic_steps = np.hstack(electronic_steps)
    accept_id = np.where(electronic_steps < nelm)[0]
    reject_id = np.where(electronic_steps >= nelm)[0]
    if len(accept_id) < trj.nframes:
        logging.info(f"skip unconverged step {reject_id}")
    trj.filter_frames(accept_id)

    try:
        accept_id = data_filter(trj)
        trj.filter_frames(accept_id)
    except Exception as e:
        logging.error(f"{e}")
        logging.error(
            "extxyz only accept batch filter work on paddedtrajectory")
        raise RuntimeError("")

    trj.name = filename

    logging.info(f"convert {filename} to {repr(trj)}")

    return trj


def write(name, trj):
    if isinstance(trj, Trajectory) and not isinstance(trj, PaddedTrajectory):
        for i in range(trj.nframes):
            natom = trj.natom[i]
            structure = Atoms(cell=trj.cells[i].reshape([3, 3]),
                              symbols=trj.species[:natom],
                              positions=trj.positions[i][:natom].reshape(
                                  [-1, 3]),
                              pbc=True)
            write_vasp(f"{i}_{name}", structure, vasp5=True)
    elif isinstance(trj, PaddedTrajectory):
        for i in range(trj.nframes):
            structure = Atoms(cell=trj.cells[i].reshape([3, 3]),
                              symbols=trj.symbols[i],
                              positions=trj.positions[i].reshape([-1, 3]),
                              pbc=True)
            write_vasp(f"{i}_{name}", structure, vasp5=True)
    else:
        raise NotImplementedError("")
