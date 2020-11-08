"""
parse CP2K output to  paddedtrajectory

type 1: cp2k_shell log, generated by ASE, with "out" in the filename
type 2: ENERGY_FORCE type, input file and xyz force output still exits
type 3:
"""
import logging

import numpy as np

from glob import glob
from os.path import isfile

from fmeee.parsers.monty import read_pattern, read_table_pattern
from fmeee.routines.folders import find_folders, find_folders_matching
from fmeee.trajectory import Trajectory, PaddedTrajectory

HARTREE = 27.2114
HARTREE_BOHR = 51.42208619083232

fl_num = r"([+-]?\d+.\d+[eE]?[+-]?\d*)"
sfl_num = r"\s+([+-]?\d+.\d+[eE]?[+-]?\d*)"
nc_fl_num = r"[+-]?\d+.\d+[eE]?[+-]?\d*"
nc_sfl_num = r"\s+[+-]?\d+.\d+[eE]?[+-]?\d*"

def get_childfolders(path):
    return find_folders_matching(['*.xyz', '*.inp', '*out*'], path)

def pack_folder_trj(folder, data_filter):

    has_xyz = len(glob(f"{folder}/*.xyz")) >0
    has_fxyz = len(glob(f"{folder}/*.xyz")) >0
    has_out = len(glob(f"{folder}/*out*")) > 0
    has_inp = (len(glob(f"{folder}/*.inp")) > 0) or (len(glob(f"{folder}/*.restart"))>0)

    has_pair = (len(glob(f"{folder}/*.inp")) > 0) and (has_out or has_xyz)
    has_xyzs = (len(glob(f"{folder}/*-frc*.xyz")) >0) and \
        (len(glob(f"{folder}/*-pos*.xyz")) >0)

    conditions=[has_xyzs, has_pair, has_out]

    trj = PaddedTrajectory()

    if np.sum(conditions) == 0:
        return trj

    # first identify force_eval pair
    MD_xyz = {}
    if has_out:

        for outfile in glob(f"{folder}/*out*"):

            if not isfile(outfile):
                continue

            outfile_dict = parse_std_out(folder, outfile)
            _trj = Trajectory()

            if 'send' in outfile_dict:
                logging.info(f"parsing {outfile} as shell product")
                _trj = parse_ase_shell_out(folder, outfile)

            elif 'run_type' in outfile_dict:
                run_type = outfile_dict['run_type']
                project_name = outfile_dict['project_name']
                if run_type == 'ENERGY_FORCE':
                    logging.info(f"parsing {outfile} as force_eval type product")
                    _trj = parse_force_eval_pairs(folder, outfile, outfile_dict)
                elif run_type == 'MD':
                    if 'xyzout' in outfile_dict:
                        MD_xyz[project_name] = (outfile, outfile_dict['xyzout'])
                    else:
                        raise NotImplementedError(f"cannot parse MD without xyz {run_type}")
                else:
                    raise NotImplementedError(f"cannot parse RUN_TYPE {run_type}")

            if _trj.nframes > 0:

                logging.info(f"hello {_trj}")
                logging.info(f"repr {repr(_trj)}")
                trj.add_trj(_trj)

    for k in MD_xyz:
        parse_md(folder, MD_xyz[k][1], MD_xyz[k][0])

    logging.info(trj)
    logging.info(repr(trj))

    return trj


    # elif np.sum(conditions) == 0:

    #     logging.info(f"! {folder} skip for no file matching")

    # else:
    #     logging.info(f"! {folder} skip for incomplete files")

    # if folder == "./":
    #     folder = "."

    # return Trajectory()

def parse_md(folder, outfile, outfile_dict, xyzfile):

    trj = PaddedTrajectory()
    trj.per_frame_attrs += ['forces', 'energies', 'positions', 'symbols']

    symbol, force = parse_forceeval_force(outfile)

    # if above strings are found
    find_input = False
    try:
        inp = outfile_dict['inputfile']
        inp = f"{folder}/{inp}"
        run_type = outfile_dict['run_type']
        proj_name = outfile_dict['project_name']
        find_input = (run_type == 'ENERGY_FORCE') and isfile(inp)
    except Exception as e:
        logging.info(f"{outfile} is not a force_eval {e}")

    if not find_input:
        return trj

    metadata = parse_std_inp(inp)
    data = parse_std_inp_pos(inp)
    data.update(metadata)

    if symbol is None:
        find_force = False
        for name in metadata['filenames']:
            for filename in glob(f"{folder}/{proj_name}*{name}*.xyz"):
                if not find_force:
                    _symbol, _force = parse_forceeval_force(filename)
                    if _symbol is not None:
                        if np.equal(_symbol, data['species']):
                            symbol = _symbol
                            force = _force
                            find_force = True
    else:
        find_force = True

    if find_force:
        data['forces'] = force.reshape([1, -1, 3])
        if 'energy' in outfile_dict:
            data['energies'] = [outfile_dict['energy']]

        _trj = Trajectory.from_dict(data)
        trj.add_trj(_trj)

    return trj

def parse_std_out(folder, outfile):

    d = read_pattern(outfile, {'inputfile':r"Input file name\s+(\S+)",
                               'run_type':r"Run type\s+(\S+)",
                               'project_name':r"Project name\s+(\S+)",
                               'xyzout':r'Coordinates\s+\d+\s+(\S+)',
                               'send':r"Sending: (GET_E)",
                               'receive':r"Received: * (READY)",
                               'energy':r"Total energy:"+sfl_num})
    del_keys = []
    for k in d:
        d[k] = np.array(d[k], dtype=str).reshape([-1])
        if len(d[k]) > 0:
            d[k] = d[k][0]
        else:
            del_keys += [k]
    for k in del_keys:
        del d[k]
    if 'energy' in d:
        d['energy'] = float(d['energy'])*HARTREE
    return d

def parse_force_eval_pairs(folder, outfile, outfile_dict):

    trj = PaddedTrajectory()
    trj.per_frame_attrs += ['forces', 'energies', 'positions', 'symbols']

    symbol, force = parse_forceeval_force(outfile)

    # if above strings are found
    find_input = False
    try:
        inp = outfile_dict['inputfile']
        inp = f"{folder}/{inp}"
        run_type = outfile_dict['run_type']
        proj_name = outfile_dict['project_name']
        find_input = (run_type == 'ENERGY_FORCE') and isfile(inp)
    except Exception as e:
        logging.info(f"{outfile} is not a force_eval {e}")

    if not find_input:
        return trj

    metadata = parse_std_inp(inp)
    data = parse_std_inp_pos(inp)
    data.update(metadata)

    if symbol is None:
        find_force = False
        for name in metadata['filenames']:
            for filename in glob(f"{folder}/{proj_name}*{name}*.xyz"):
                if not find_force:
                    _symbol, _force = parse_forceeval_force(filename)
                    if _symbol is not None:
                        if np.equal(_symbol, data['species']):
                            symbol = _symbol
                            force = _force
                            find_force = True
    else:
        find_force = True

    if find_force:
        data['forces'] = force
        if 'energy' in outfile_dict:
            data['energies'] = [outfile_dict['energy']]

        trj = Trajectory.from_dict(data)


    return trj


def parse_forceeval_force(filename):

    header_pattern=r"\#\s+Atom\s+Kind\s+Element\s+X\s+Y\s+Z"
    footer_pattern=r"SUM OF ATOMIC FORCES\s+"+nc_sfl_num*4

    d = read_pattern(filename, {'header':r"\#\s+Atom\s+Kind\s+Element\s+X\s+Y\s+(Z)",
                               'footer':r"SUM OF ATOMIC FORCES\s+"+sfl_num*4})
    if len(d['footer']) > 0:
        force = read_table_pattern(filename,
                                   row_pattern=r"\d+\s+\d+\s+([A-Z][a-z]*?)" + \
                                       sfl_num*3,
                                   header_pattern=header_pattern,
                                   footer_pattern=footer_pattern,
                                   last_one_only=False
                                   )
        if len(force) > 0:
            force = np.array(force[0], str)
            symbol = force[:, 0]
            force = np.array(force[:, 1:], dtype=float).reshape([1, -1, 3])*HARTREE_BOHR
            return symbol, force
    return None, None

def parse_std_inp_pos(filename):

    data = {}

    position = read_table_pattern(filename,
                             header_pattern=r"\&COORD",
                             row_pattern=r"([A-Z][a-z]*?)" + \
                                 sfl_num*3,
                             footer_pattern=r"\s+\&END COORD\s+",
                             last_one_only=False
                             )
    position = np.array(position[0], str)
    data['species'] = position[:, 0].reshape([-1])
    data['positions'] = np.array(position[:, 1:], dtype=float).reshape([1, -1, 3])
    data['natom'] = data['positions'].shape[1]

    cell = read_table_pattern(filename,
                              header_pattern=r"\&CELL",
                              row_pattern=r"[A-Ca-c]" + sfl_num*3,
                              footer_pattern=r"&END",
                              last_one_only=False
                              )
    if len(cell) > 0:
        data['cells'] = np.array(cell[0], dtype=float).reshape([1, 3, 3])

    return data

def parse_std_inp_metadata(filename):

    data = {}

    d = read_pattern(filename,
                     {'kpoints':r"SCHEME MONKHORST-PACK\s(\d+)\s(\d+)\s(\d+)",
                      'gamma':r"SCHEME ([gG][a-zA-Z]*)",
                      'cutoff': r"REL_CUTOFF\s+(\d+\.*\d*)",
                      'thermostat':r"ENSEMBLE\s+(\w*)",
                      'dipole_correction':r"SURFACE_DIPOLE_CORRECTION (\w+)",
                      'run_type':r"RUN_TYPE\s+(\S+)",
                      'project_name':r"PROJECT\s+(\S+)",
                      'filenames':r'FILENAME (\S+)'})

    if len(d['kpoints']) > 0:
        data['kpoints'] = [int(i) for i in d['kpoints'][-1]]

    if len(d['gamma']) > 0:
        data['gamma'] = True

    if len(d['cutoff']) > 0:
        data['cutoff'] = float(d['cutoff'][-1][0])

    if len(d['thermostat']) > 0:
        data['thermostat'] = d['thermostat'][-1][0]
        data['aimd'] = True
    else:
        data['aimd'] = False

    if len(d['dipole_correction']) > 0:
        data['dipole_correction'] = True
    else:
        data['dipole_correction'] = False

    data['filenames'] = np.array(d['filenames'], dtype=str).reshape([-1])

    return data

    # 'fix-atoms': false,
    # 'non-fix-atoms': '$(grep -i list $file|awk '{print $2}')',
    # 'meltedCu':false,
    # 'started from fix bottom': true,
#     'mass': [$(grep -i mass $file|awk '{printf "%5.2f,", $2}') ],
#     'timestep': $(grep -i timestep $file|awk '{printf "%3.1f:", $2}') ,
# },

def parse_ase_shell_out(folder, filename):
    """
    assume all frames share the same symbols
    """

    trj = Trajectory()

    with open(filename) as fin:
        lines = fin.readlines()

    nlines = len(lines)

    nconfigs = 0
    i = 0
    cell = None
    position = None
    energy = None
    force = None
    input_dict = None
    species = None
    data = {}

    while (i<nlines):

        if "LOAD" in lines[i]:
            inputfile=lines[i].split()[2]

        elif "SET_CELL" in lines[i]:
            cell  = []
            for icell in range(3):
                i+= 1
                cell_line = lines[i].split()[1:]
                cell += [[float(x) for x in cell_line]]
            cell = np.array(cell).reshape([-1])

        elif "SET_POS" in lines[i]:
            i += 1
            natom = int(lines[i].split()[1])//3
            position = []
            for iatom in range(natom):
                i += 1
                l = lines[i].split()
                pos_line = l[1:]
                position += [[float(x) for x in pos_line]]

            if input_dict is None and isfile(inputfile):
                input_dict = parse_std_inp_metadata(inputfile)
                data.update(input_dict)

                d = parse_std_inp_pos(inputfile)
                species = d['species']

        elif "GET_E" in lines[i]:
            i += 1
            energy = float(lines[i].split()[1])

        elif "GET_F" in lines[i]:

            i += 1
            natom = int(lines[i].split()[1])//3
            force = []
            for iatom in range(natom):
                i += 1
                force_line = lines[i].split()[1:]
                force += [[float(x) for x in force_line]]
            force = np.array(force).reshape([-1])

            position = np.array(position).reshape([-1])

            data['cells'] = np.copy(cell).reshape([1, 3, 3])
            # data['species'] = species
            data['positions'] = np.copy(position).reshape([1, -1, 3])
            data['energies'] = [energy]
            data['forces'] = np.copy(force).reshape([1, -1, 3])
            data['natom'] = natom

            _trj = Trajectory.from_dict(data)
            trj.add_trj(_trj)

        i += 1

    trj.species = species

    return trj
