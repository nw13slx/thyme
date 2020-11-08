"""
parse CP2K output to  paddedtrajectory

type 1: cp2k_shell log, generated by ASE, with "out" in the filename
type 2: ENERGY_FORCE type, input file and xyz force output still exits
type 3:
"""
import logging

import numpy as np

from glob import glob
from os.path import isfile, getctime

from fmeee.parsers.monty import read_pattern, read_table_pattern
from fmeee.routines.folders import find_folders, find_folders_matching
from fmeee.trajectory import Trajectory, PaddedTrajectory

HARTREE = 27.2114
HARTREE_BOHR = 51.42208619083232

nc_fl_num = r"[+-]?\d+\.*\d*[eE]?[+-]?\d*"
fl_num = r"("+nc_fl_num+")"
nc_sfl_num = r"\s+"+nc_fl_num
sfl_num = r"\s+"+fl_num

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

    trj = Trajectory()

    if np.sum(conditions) == 0:
        return trj

    # first identify force_eval pair
    MD_xyz = {}
    if has_out:

        for outfile in glob(f"{folder}/*out*"):

            if not isfile(outfile):
                continue

            outfile_dict = parse_std_out(folder, outfile)
            if 'abort' in outfile_dict:
                continue

            _trj = Trajectory()

            if 'send' in outfile_dict:
                logging.info(f"parsing {outfile} as shell product")
                _trj = parse_ase_shell_out(folder, outfile)

            elif 'run_type' in outfile_dict:
                run_type = outfile_dict['run_type']
                proj_name = outfile_dict['proj_name']
                if run_type == 'ENERGY_FORCE':
                    logging.info(f"parsing {outfile} as force_eval type product")
                    _trj = parse_force_eval_pairs(folder, outfile, outfile_dict)
                elif run_type == 'MD':
                    if 'xyzout' in outfile_dict:
                        MD_xyz[proj_name] = (outfile, outfile_dict['xyzout'])
                    else:
                        raise NotImplementedError(f"cannot parse MD without xyz {run_type}")
                else:
                    raise NotImplementedError(f"cannot parse RUN_TYPE {run_type}")

            if _trj.nframes > 0:
                logging.info(f"repr {repr(_trj)}")
                logging.info(f"add {_trj}")
                logging.info(f"to {trj}")
                trj.add_trj(_trj)

    for k in MD_xyz:
        _trj = parse_md(folder, outfile_dict=outfile_dict)
        logging.info(f"repr {repr(_trj)}")
        trj.add_trj(_trj)

    if has_xyz and has_inp and len(MD_xyz) == 0:

        mtime = 0
        mfile = ""

        for filename in glob(f"{folder}/*.inp") + glob(f"{folder}/*.restart"):
            if getctime(filename) > mtime:
                mtime = getctime(filename)
                mfile = filename

        outfile_dict=dict(run_type="MD",
                          inputfile=mfile
                          )


        for filename in glob("*-frc*.xyz"):
            proj_name = filename.split("-")[0]
            outfile_dict['proj_name'] = proj_name
            if isfile(f"{proj_name}-pos-1.xyz"):
                _trj = parse_md(folder, outfile_dict=outfile_dict)
                logging.info(f"repr {repr(_trj)}")
                trj.add_trj(_trj)

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

def parse_md(folder, outfile_dict):

    logging.info(f"parse md in folder {folder}")

    # if above strings are found
    find_input = False
    try:
        inp = outfile_dict['inputfile']
        inp = f"{folder}/{inp}"
        run_type = outfile_dict['run_type']
        proj_name = outfile_dict['proj_name']
        find_input = (run_type == 'MD') and isfile(inp)
    except Exception as e:
        logging.info(f"It is not a MD {e}")

    if not find_input:
        return Trajectory()

    metadata = parse_std_inp_metadata(inp)
    data = parse_std_inp_pos(inp)

    return parse_cp2k_xyzs(f"{folder}/{proj_name}-pos-1.xyz",
                           f"{folder}/{proj_name}-frc-1.xyz",
                           data['cells'], metadata)


def parse_std_out(folder, outfile):

    logging.info(f"parse {outfile}")

    d = read_pattern(outfile, {'inputfile':r"Input file name\s+(\S+)",
                               'abort':r"(ABORT)",
                               'run_type':r"Run type\s+(\S+)",
                               'proj_name':r"Project name\s+(\S+)",
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

    logging.info(f"parse {outfile}")

    trj = Trajectory()
    trj.per_frame_attrs += ['forces', 'energies', 'positions'] #, 'symbols']

    symbol, force = parse_forceeval_force(outfile)

    # if above strings are found
    find_input = False
    try:
        inp = outfile_dict['inputfile']
        inp = f"{folder}/{inp}"
        run_type = outfile_dict['run_type']
        proj_name = outfile_dict['proj_name']
        find_input = (run_type == 'ENERGY_FORCE') and isfile(inp)
    except Exception as e:
        logging.info(f"{outfile} is not a force_eval {e}")

    if not find_input:
        return trj

    metadata = parse_std_inp_metadata(inp)
    data = parse_std_inp_pos(inp)
    data.update(metadata)

    if symbol is None:
        find_force = False
        for name in metadata['filenames']:
            for filename in glob(f"{folder}/{proj_name}*{name}*.xyz"):
                if not find_force:
                    _symbol, _force = parse_forceeval_force(filename)
                    if _symbol is not None:
                        if all(_symbol == data['species']):
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

    logging.info(f"parse {filename}")

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

    logging.info(f"parse {filename}")

    data = {}

    if 'restart' in filename:
        footer = r"\s+UNIT angstrom"
    else:
        footer=r"\s+\&END COORD\s+"

    position = read_table_pattern(filename,
                             header_pattern=r"\&COORD",
                             row_pattern=r"([A-Z][a-z]*?)" + \
                                 sfl_num*3,
                             footer_pattern=footer,
                             last_one_only=False
                             )
    try:
        position = np.array(position[0], str)
        data['species'] = position[:, 0].reshape([-1])
        data['positions'] = np.array(position[:, 1:], dtype=float).reshape([1, -1, 3])
        data['natom'] = data['positions'].shape[1]
    except:
        pass

    if 'restart' in filename:
        footer=r"MULTIPLE_UNIT_CELL"
    else:
        footer=r"&END"
    cell = read_table_pattern(filename,
                              header_pattern=r"\&CELL",
                              row_pattern=r"[A-Ca-c]" + sfl_num*3,
                              footer_pattern=footer,
                              last_one_only=False
                              )
    if len(cell) > 0:
        data['cells'] = np.array(cell[0], dtype=float).reshape([1, 3, 3])

    return data

def parse_std_inp_metadata(filename):

    logging.info(f"parse {filename}")

    data = {}

    d = read_pattern(filename,
                     {'kpoints':r"SCHEME MONKHORST-PACK\s(\d+)\s(\d+)\s(\d+)",
                      'gamma':r"SCHEME ([gG][a-zA-Z]*)",
                      'cutoff': r"REL_CUTOFF"+sfl_num,
                      'thermostat':r"ENSEMBLE\s+(\w*)",
                      'dipole_correction':r"SURFACE_DIPOLE_CORRECTION (\w+)",
                      'run_type':r"RUN_TYPE\s+(\S+)",
                      'proj_name':r"PROJECT\s+(\S+)",
                      'filenames':r'FILENAME (\S+)',
                      'etemp':r'ELECTRONIC_TEMPERATURE [K] (\w+)'})

    fix = read_table_pattern(filename,
                             header_pattern=r"\&FIXED_ATOMS",
                             row_pattern=r"LIST\s+(\d+)\.\.(\d+)",
                             footer_pattern=r"&END",
                             last_one_only=False
                             )

    if len(fix) > 0:
        fix = np.arange(int(fix[0]), int(fix[1])+1)
        data['fix_atoms'] = True
        data['fix_atoms_id'] = fix
    else:
        data['fix_atoms'] = False

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

    if len(d['etemp']) > 0:
        data['etemp'] = float(d['etemp'][0][0])

    data['filenames'] = np.array(d['filenames'], dtype=str).reshape([-1])

    return data

    # 'meltedCu':false,
    # 'started from fix bottom': true,
#     'mass': [$(grep -i mass $file|awk '{printf "%5.2f,", $2}') ],
#     'timestep': $(grep -i timestep $file|awk '{printf "%3.1f:", $2}') ,
# },

def parse_ase_shell_out(folder, filename):
    """
    assume all frames share the same symbols
    """
    logging.info(f"parse {filename}")

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

    for filename in glob(f"{folder}/*.inp"):
        if species is None:
            data = parse_std_inp_pos(filename)
            species = data['species']

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

    if species is not None:
        trj.species = np.array(species, dtype=str).reshape([-1])
        if 'species' not in trj.metadata_attrs:
            trj.metadata_attrs += ['species']
    else:
        logging.info(f"{trj}")
        logging.info(f"cannot find species, give up the whole frame")
        return Trajectory()

    logging.debug(f"shell species {trj.species}")
    return trj

def parse_cp2k_xyzs(posxyz, forcexyz, cell, metadata):

    logging.info(f"parse {posxyz} {forcexyz}")

    d = \
        read_pattern(posxyz,
                     {'natoms':r"^\s*([0-9]+)\s*$",
                      'energies':r"E\s*=\s*"+fl_num,
                      'pos': r"^\s*[A-Z][a-zA-Z]*"+sfl_num*3,
                      'symbols': r"^\s*([A-Z][a-zA-Z]*)"+nc_sfl_num*3,
                      })

    d_f = \
        read_pattern(forcexyz,
                     {'pos': r"^\s*[A-Z][a-zA-Z]*"+sfl_num*3,
                      'symbols': r"^\s*([A-Z][a-zA-Z]*)"+nc_sfl_num*3,
                      })

    nframes = len(d['energies'])
    if nframes == 0:
        return Trajectory()

    dictionary = dict(
        positions = np.array(d['pos'], dtype=float).reshape([nframes, -1, 3]),
        forces = np.array(d_f['pos'], dtype=float).reshape([nframes, -1, 3])*HARTREE_BOHR,
        energies = np.array(d['energies'], dtype=float).reshape([-1])*HARTREE,
        cells = np.array([cell]*nframes, dtype=float).reshape([nframes, 3, 3])
    )

    # double check all arrays have the same number of frames
    nframes = []
    for k in dictionary:
        if k!= 'natom':
            nframes += [dictionary[k].shape[0]]
    assert len(set(nframes)) == 1

    natom=int(d['natoms'][0][0])
    dictionary.update(metadata)
    dictionary['species'] = np.array(d['symbols'][:natom], dtype=str).reshape([-1])

    trj = Trajectory.from_dict(dictionary)

    return trj
