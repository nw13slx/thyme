import logging

import numpy as np
import time

from thyme.parsers.monty import read_pattern, read_table_pattern
from thyme.routines.folders import find_folders
from thyme.trajectory import Trajectory
from thyme._key import *

from collections import Counter
from os.path import isfile

from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar
from ase.io.vasp import write_vasp
from ase.atoms import Atoms


def get_childfolders(path):

    return find_folders(["vasprun.xml", "OUTCAR", "vasp.out"], path)


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
        data["kpoints"] = kpoints.kpts[0]

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
                data["species"] = species
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
    if not isfile(filename):
        return Trajectory()

    t = time.time()
    d_total_energy = read_pattern(
        filename,
        {"total_energy": r"free  energy   TOTEN\s+=\s+([+-]?\d+.\d+)"},
        postprocess=lambda x: float(x),
    )
    if len(d_total_energy["total_energy"]) == 0:
        return Trajectory()
    total_energy = np.hstack(d_total_energy["total_energy"])

    logging.info(f" parsing {filename} for positions")
    pos_force = read_table_pattern(
        filename,
        header_pattern=r"\sPOSITION\s+TOTAL-FORCE \(eV/Angst\)\n\s-+",
        row_pattern=r"^\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)$",
        footer_pattern=r"\s--+",
        postprocess=lambda x: float(x),
        last_one_only=False,
    )
    pos_force = np.array(pos_force, dtype=np.float64)

    logging.info(f" parsing {filename} for cells")
    cells = read_table_pattern(
        filename,
        header_pattern=r"\sdirect lattice vectors\s+reciprocal lattice vectors",
        row_pattern=r"\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)"
        "\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)",
        footer_pattern=r"^$",
        postprocess=lambda x: float(x),
        last_one_only=False,
    )
    cells = np.array(cells, dtype=np.float64)

    d_n_e_iter = read_pattern(
        filename,
        {"n_e_iter": r"Iteration\s+(\d+)\s?\(\s+\d+\)"},
        postprocess=lambda x: int(x),
    )
    n_electronic_steps = np.array(d_n_e_iter["n_e_iter"], dtype=int).reshape([-1])
    logging.info(f"Outcar grep time {time.time()-t}")
    logging.info("loaded outcar")

    t = time.time()
    try:
        incar = Incar.from_file(filename)
        data["nelm"] = incar["NELM"]
        data["cutoff"] = incar["ENCUT"]
        data["dipole_correction"] = bool(incar["LDIPOL"].split()[0])
        data.update(incar)
        logging.info(f"Incar grep time {time.time()-t}")
    except Exception as e:
        logging.info(f"fail to load incar {e}")
        return Trajectory()

    nelm = data["nelm"]

    cs = cells[:, :, :3].reshape([-1, 3, 3])

    # remove unconverged ionic steps
    c = Counter(n_electronic_steps)
    n_electronic_steps = [c[k] for k in sorted(c.keys())]
    nframes = pos_force.shape[0]
    converged_steps = np.array(
        [i for i, s in enumerate(n_electronic_steps) if (s < nelm and i < nframes)]
    )
    # log and return tempty trajectory if needed
    if len(converged_steps) == 0:
        return Trajectory()

    elif len(converged_steps) < nframes:
        logging.info(
            "skip unconverged step {}".format(
                [
                    i
                    for i, s in enumerate(n_electronic_steps)
                    if (s >= nelm or i >= nframes)
                ]
            )
        )

    data.update(
        {
            CELL: cs[converged_steps],
            POSITION: pos_force[converged_steps, :, :3],
            FORCE: pos_force[converged_steps, :, 3:],
            TOTAL_ENERGY: total_energy[converged_steps],
            SPECIES: species,
            "electronic_steps": np.hstack(n_electronic_steps)[converged_steps],
            PER_FRAME_ATTRS: [POSITION, FORCE, TOTAL_ENERGY, CELL, "electronic_steps"],
            FIXED_ATTRS: [SPECIES, NATOMS],
        }
    )

    trj = Trajectory.from_dict(data)

    accept_id = data_filter(trj)
    trj.include_frames(accept_id)

    trj.name = filename

    logging.info(f"convert {filename} to {repr(trj)}")
    logging.info(f"{trj}")

    return trj


def parse_vasprun_trj(folder, data_filter):

    data = {}
    filename = "/".join([folder, "vasprun.xml"])
    if not isfile(filename):
        return Trajectory()

    try:
        vasprun = Vasprun(filename, ionic_step_skip=0, exception_on_bad_xml=False)
    except Exception as e:
        logging.info(f"fail to load vasprun {e}")
        return Trajectory()

    nelm = vasprun.incar["NELM"]
    data["nelm"] = nelm
    data["cutoff"] = vasprun.incar["ENCUT"]
    data["dipole_correction"] = vasprun.parameters["LDIPOL"]
    species = vasprun.atomic_symbols
    data["kpoints"] = vasprun.kpoints.kpts[0]

    positions = []
    forces = []
    total_energy = []
    cells = []
    electronic_steps = []
    for istep, step in enumerate(vasprun.ionic_steps):
        es = len(step["electronic_steps"])
        if es < nelm:
            electronic_steps += [es]
            positions += [step["structure"].cart_coords.reshape([-1])]
            forces += [np.hstack(step["forces"])]
            total_energy += [step["e_fr_energy"]]
            cells += [step["structure"].lattice.matrix.reshape([-1])]
        else:
            logging.info(f"skip unconverged step {istep}")

    if len(electronic_steps) == 0:
        return Trajectory()

    data.update(
        {
            CELL: np.vstack(cells),
            POSITION: np.vstack(positions),
            FORCE: np.vstack(forces),
            TOTAL_ENERGY: np.hstack(total_energy),
            SPECIES: species,
            "electronic_steps": np.hstack(electronic_steps),
            PER_FRAME_ATTRS: [POSITION, FORCE, TOTAL_ENERGY, CELL, "electronic_steps"],
            FIXED_ATTRS: [SPECIES, NATOMS],
        }
    )
    trj = Trajectory.from_dict(data)

    accept_id = data_filter(trj)
    trj.include_frames(accept_id)

    trj.name = filename

    logging.info(f"convert {filename} to {repr(trj)}")

    return trj


def write(name, trj):
    for i in range(trj.nframes):
        frame = trj.get_frame(i)
        definition = {"pbc": False}
        if CELL in frame:
            definition["cell"] = frame[CELL]
            definition["pbc"] = True
        structure = Atoms(
            symbols=frame["species"], positions=frame["position"], **definition
        )
        write_vasp(f"{name}_{i}.poscar", structure, vasp5=True)


def compare_metadata(trj1, trj2):
    keys = [
        "DEG_THRESHOLD",
        "DFIELD",
        "EDIFF",
        "EDIFFG",
        "ENAUG",
        "ENCUT",
        "EPSILON",
        "GGA",
        "GGA_COMPAT",
        "IALGO",
        "IDIPOL",
        "IRESTART",
        "ISMEAR",
        "ISPIN",
        "IVDW",
        "KBLOCK",
        "kpoints",
        "LASPH",
        "LCHIMAG",
        "LCORR",
        "LDIAG",
        "LDIPOL",
        "LDOWNSAMPLE",
        "LHFCALC",
        "LINTERFAST",
        "LMONO",
        "LRPA",
        "LSORBIT",
        "LVDW_EWALD",
        "LVDW_ONECELL",
        "LVEL",
        "LVHAR",
        "LWAVE",
        "METAGGA",
        "PREC",
        "PSTRESS",
        "SCALEE",
        "SIGMA",
        "TEEND",
        "VDW_D",
        "VDW_IDAMPF",
        "VDW_RADIUS",
        "VDW_S6",
        "VDW_SR",
        "VOSKOWN",
    ]
    meta1 = trj1.metadata_attrs
    meta2 = trj2.metadata_attrs
    for key in keys:
        if key in meta1 and key in meta2:
            item1 = getattr(trj1, key)
            item2 = getattr(trj2, key)
            if item1 != item2:
                logging.debug(f"{key} does not match, thus not merging")
                return False
        elif key not in meta1 and key not in meta2:
            pass
        else:
            logging.info(f"{key}")
            return False
    return True
