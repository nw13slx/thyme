import logging
logging.basicConfig(filename=f'collect.log', filemode='w',
                    level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

import json
import numpy as np
import time

import analysis.reduced_outcar as reduced_outcar

from ase.atoms import Atoms
from collections import Counter
from glob import glob
from os import walk, mkdir
from os.path import dirname, join, basename, isdir, isfile

from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar

def e_filter(xyz, f, e, c, species):
    atoms = Atoms(species, xyz, cell=c, pbc=True)
    dist_mat = atoms.get_all_distances(mic=True)
    not_Au = [i for i in range(dist_mat.shape[0]) if species[i] != 'Au']
    for ind in range(dist_mat.shape[0]):

        neigh = np.argmin(np.hstack([dist_mat[ind, :ind], dist_mat[ind, ind+1:]]))
        if neigh >= ind:
            neigh += 1
        mindist = dist_mat[ind, neigh]

        if ind in not_Au and mindist > 2.0:
            logging.info(f"skip frame for isolated CHO atom {mindist} {ind} {species[ind]} {neigh} {species[neigh]}")
            return False
        elif ind not in not_Au:
            # if mindist > 3.0:
            #     logging.info(f"skip frame for isolated Au atom {mindist} {ind} {species[ind]} {neigh} {species[neigh]}")
            #     return False
            Au = [i for i in range(dist_mat.shape[0]) if species[i] == 'Au' and i!=ind]
            neigh = Au[np.argmin(dist_mat[ind, Au])]
            minAu = dist_mat[ind, neigh]
            if minAu > 3.5:
                print(ind, dist_mat[ind, ind-3:np.min([ind+3, dist_mat.shape[0]])])
                logging.info(f"skip frame for isolated Au from other Au {minAu} {ind} {species[ind]} {neigh} {species[neigh]}")
                return False


    return True
    # C_id = np.array([index for index, ele in enumerate(species) if ele == 'C'])
    # for Cindex in C_id:
    #     if xyz[Cindex, 2] < 10:
    #         logging.info("skip frame for low C")
    #         return False
    # if (e+411 < -100) or (e+411 > 40):
    #     logging.info(f"skip frame for high/low e {e+411:8.2f}")
    #     return False
    # return True

def main():

    # mkfolder("npz")
    folders = search_all_folders(['vasprun.xml', 'OUTCAR', 'vasp.out'])
    folders = sorted(folders)
    logging.info(f"all folders: {folders}")

    count = 0
    alldata = {}
    for folder in folders:

        if folder == "./":
            casename = "current_folder"
        if folder[:2] == "./":
            casename = "_".join(folder[2:].split("/"))
        else:
            casename = "_".join(folder.split("/"))

        logging.info(casename)

        data = pack_folder(folder, e_filter)
        if data['nframes'] >= 1:
            logging.info(f"{folder}, {casename}, {data['nframes']}")
            alldata[casename] = data
            count += 1
            if count%10 == 0:
                np.savez("alldata.npz", **alldata)
        else:
            logging.info(f"! skip whole folder {casename}, {data['nframes']}")

    np.savez("alldata.npz", **alldata)
    logging.info("Complete")

def search_all_folders(filenames):
    folders = find_folders(filenames, "./")
    return folders

def find_folders(filenames, path):

    result = set([root \
                  for root, dirs, files in walk(path) \
                  if len((set(files)).intersection(filenames)) >0])
    return result

def mkfolder(name):
    if not isdir(name):
        mkdir(name)

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
        reduced_outcar.read_pattern(filename, {'energies':r"free  energy   TOTEN\s+=\s+([+-]?\d+.\d+)"}, postprocess=lambda x:float(x))
    if len(d_energies['energies']) == 0:
        return {}
    energies = np.hstack(d_energies['energies'])

    pos_force = reduced_outcar.read_table_pattern(filename,
        header_pattern=r"\sPOSITION\s+TOTAL-FORCE \(eV/Angst\)\n\s-+",
        row_pattern=r"\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)",
        footer_pattern=r"\s--+",
        postprocess=lambda x: float(x),
        last_one_only=False

    )
    pos_force = np.array(pos_force, dtype=np.float64)

    cells = reduced_outcar.read_table_pattern(filename,
        header_pattern=r"\sdirect lattice vectors\s+reciprocal lattice vectors",
        row_pattern=r"\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)"
        "\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)",
        footer_pattern=r"^$",
        postprocess=lambda x: float(x),
        last_one_only=False
    )
    cells = np.array(cells, dtype=np.float64)

    d_n_e_iter = reduced_outcar.read_pattern(filename,
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


if __name__ == '__main__':
    main()
