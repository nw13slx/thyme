import logging

import numpy as np
import time

from fmeee.parsers.monty import read_pattern, read_table_pattern
from fmeee.routines.folders import find_folders

from collections import Counter
from os.path import isfile

from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar

def get_childfolders(path):

    return find_folders(['vasprun.xml', 'OUTCAR', 'vasp.out'], path)

def pack_folder(folder, data_filter):

    data = dict(
        positions = None,
        forces = None,
        energies = None,
        cells = None,
        nelm = None,
        nframes = 0,
        cutoff = None,
        dipole_correction = None,
        species = None,
        natom = None,
        kpoints = None
    )

    if folder == "./":
        folder = "."

    meta = parse_outcar(folder, data_filter)
    if len(meta) > 1:
        data.update(meta)
    else:
        meta = parse_vasprun(folder, data_filter)
        if len(meta) > 1:
            data.update(meta)
    return data


def parse_outcar(folder, data_filter):

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
                return {}
        # else:
        #     logging.info("cannot find", filename2)

    if species is None:
        logging.info("cannot find speciesin either POSCAR or CONTCAR")
        return {}

    filename = "/".join([folder, "OUTCAR"])

    t = time.time()
    d_energies = \
        read_pattern(filename,
                     {'energies':r"free  energy   TOTEN\s+=\s+([+-]?\d+.\d+)"},
                     postprocess=lambda x:float(x))
    if len(d_energies['energies']) == 0:
        return {}
    energies = np.hstack(d_energies['energies'])

    pos_force = read_table_pattern(filename,
        header_pattern=r"\sPOSITION\s+TOTAL-FORCE \(eV/Angst\)\n\s-+",
        row_pattern=r"\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)",
        footer_pattern=r"\s--+",
        postprocess=lambda x: float(x),
        last_one_only=False

    )
    pos_force = np.array(pos_force, dtype=np.float64)

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
                                           {'n_e_iter':r"Iteration\s+(\d+)\s?\(\s+\d+\)"},
                                            postprocess=lambda x: int(x))
    n_electronic_steps = np.array(d_n_e_iter['n_e_iter'], dtype=int).reshape([-1])
    logging.debug("grep time {time.time()-t}")

    try:
        incar = Incar.from_file(filename)
    except Exception as e:
        logging.info("fail to load outcar", e)
        return {}
    logging.info("load outcar")

    nelm = incar['NELM']
    data['nelm'] = nelm
    data['cutoff'] =incar['ENCUT']
    data['dipole_correction'] = bool(incar['LDIPOL'].split()[0])
    data.update(incar)

    cs = cells[:, :, :3].reshape([-1, 3, 3])

    # remove unconverged ionic steps
    c=Counter(n_electronic_steps)
    n_electronic_steps = [ c[k] for k in sorted(c.keys()) ]
    nframes = pos_force.shape[0]
    converged_steps = np.array([i for i, s in enumerate(n_electronic_steps) \
                                if (s < nelm and i < nframes)])
    if len(converged_steps) < nframes:
        logging.info("skip unconverged step", [i for i, s in enumerate(n_electronic_steps) \
                                        if (s >= nelm or i >= nframes)])
    if len(converged_steps) == 0:
        return data

    natom = pos_force.shape[1]
    data['natom'] = natom

    cs = cs[converged_steps]
    xyzs = pos_force[converged_steps, :, :3]
    fs = pos_force[converged_steps, :, 3:]
    es = energies[converged_steps]

    positions = []
    forces = []
    energies = []
    cells = []
    nframes = xyzs.shape[0]
    for istep in range(nframes):
        xyz = xyzs[istep]
        f = fs[istep]
        e = es[istep]
        c = cs[istep]
        if data_filter(xyz, f, e, c, species):
            positions += [xyz.reshape([-1])]
            forces += [np.hstack(f)]
            energies += [e]
            cells += [c.reshape([-1])]
    nframes = len(positions)
    if nframes >=1 :
        data['positions'] = np.vstack(positions)
        data['forces'] = np.vstack(forces)
        data['energies'] = np.hstack(energies)
        data['cells'] = np.vstack(cells)
    data['nframes'] = nframes
    data['species'] = species

    return data

def parse_vasprun(folder, data_filter):

    data = {}
    filename = "/".join([folder, "vasprun.xml"])
    if isfile(filename):
        try:
            vasprun = Vasprun(filename, ionic_step_skip=0,
                              exception_on_bad_xml=False)
        except Exception as e:
            logging.info("fail to load vasprun", e)
            return data

        nelm = vasprun.incar['NELM']
        data['nelm'] = nelm
        data['cutoff'] =vasprun.incar['ENCUT']
        data['dipole_correction'] = vasprun.parameters['LDIPOL']
        species = vasprun.atomic_symbols
        data['species'] = species
        data['natom'] = len(vasprun.atomic_symbols)
        data['kpoints'] = vasprun.kpoints.kpts[0]

        positions = []
        forces = []
        energies = []
        cells = []
        for step in vasprun.ionic_steps:
            if len(step['electronic_steps']) < nelm:
                xyz = step['structure'].cart_coords
                f = step['forces']
                e = step['e_fr_energy']
                c = step['structure'].lattice.matrix
                if data_filter(xyz, f, e, c, species):
                    positions += [xyz.reshape([-1])]
                    forces += [np.hstack(f)]
                    energies += [e]
                    cells += [c.reshape([-1])]
            else:
                logging.info("skip unconverged", len(step['electronic_steps']))
        nframes = len(positions)
        if nframes >=1 :
            data['positions'] = np.vstack(positions)
            data['forces'] = np.vstack(forces)
            data['energies'] = np.hstack(energies)
            data['cells'] = np.vstack(cells)
        data['nframes'] = nframes

    return data
