import logging

import numpy as np

from glob import glob
from os.path import isfile

from fmeee.parsers.monty import read_pattern, read_table_pattern
from fmeee.routines.folders import find_folders, find_folders_matching

def get_childfolders(path):
    return find_folders_matching(['*.xyz', '*.inp', '*out*'], path)

def pack_folder(folder, data_filter):

    data = dict(
        nframes = 0,
    )

    hasfrcxyz = len(glob(f"{folder}/*-frc*.xyz")) >0
    hasposxyz = len(glob(f"{folder}/*-pos*.xyz")) >0
    hasinp = (len(glob(f"{folder}/*.inp")) > 0) or (len(glob(f"{folder}/*.restart")))
    hasout = len(glob(f"{folder}/*out*")) > 0
    conditions=[hasfrcxyz, hasposxyz, hasinp, hasout]

    force_eval_pair = []
    cp2k_shell_out = []
    if np.sum(conditions) > 0:

        inputfiles = []
        for outfile in glob(f"{folder}/*out*"):
            if isfile(outfile) and 'xyz' not in outfile:

                # check whether it is shell pairs
                try:
                    d = read_pattern(outfile, {'inputfile':r"Input file name\s+([\S]+)",
                                                              'forces':r"(ATOMIC FORCES in)"})
                except Exception as e:
                    print(e, outfile)
                    raise RuntimeError("hello")

                if len(d['inputfile'])>0:
                    inp = f"{folder}/{d['inputfile'][0][0]}"
                    if isfile(inp):
                        inputfiles += [inp]
                        if len(d['forces'])==0:
                            find_force = 0
                            filename=None
                            for frcxyz in glob(f"{folder}/*f*.xyz"):
                                df = read_pattern(frcxyz, {'forces':r"(ATOMIC FORCES in \[a\.u\.\])"})
                                if len(df['forces'])>0:
                                    if find_force ==0:
                                        find_force += 1
                                        force_eval_pair += [[outfile, inp, frcxyz]]
                                        logging.info(f"found force eval triplets {outfile} {inp} {frcxyz}")
                                    else:
                                        find_force += 1
                                        logging.info(f"! too many force output for {outfile}")
                        else:
                            force_eval_pair += [[outfile, inp, None]]
                            logging.info(f"found force eval pairs {outfile} {inp}")

                d = read_pattern(outfile, {'load':r"LOAD\s+(\S+.inp)\s+(\S+.out)"})
                if len(d['load']) > 0:
                    logging.info(f"found cp2k_shell screen output {outfile}")
                    cp2k_shell_out += [outfile]


    if len(force_eval_pair) > 0:
        logging.info(f"{folder} force_eval {len(force_eval_pair)}")
        data.update(parse_force_eval(folder, force_eval_pair, data_filter))
    elif len(cp2k_shell_out) > 0:
        logging.info(f"{folder} cp2k_shell {len(cp2k_shell_out)}")
        data.update(parse_cp2k_shell(folder, sorted(cp2k_shell_out), data_filter))
    elif hasfrcxyz and hasposxyz:
        logging.info(f"{folder} MD simulations")
        # data = parse_cp2k_md(folder, inputfiles, data_filter)
    elif np.sum(conditions) == 0:
        logging.info(f"! {folder} skip for no file matching")
    else:
        logging.info(f"! {folder} skip for incomplete files")

    if folder == "./":
        folder = "."

    return data

def parse_cp2k_inp(filename):

    data = {}

    d = read_pattern(filename,
                     {'kpoints':r"SCHEME MONKHORST-PACK\s(\d+)\s(\d+)\s(\d+)",
                      'gamma':r"SCHEME ([gG][a-zA-Z]*)",
                      'cutoff': r"REL_CUTOFF\s+(\d+\.*\d*)",
                      'thermostat':r"ENSEMBLE\s+(\w*)",
                      'dipole_correction':r"SURFACE_DIPOLE_CORRECTION ([\w]+)"})
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

    return data


    # 'input_filename': $file,
    # 'fix-atoms': false,
    # 'non-fix-atoms': '$(grep -i list $file|awk '{print $2}')',
    # 'meltedCu':false,
    # 'aimd': true,
    # 'started from fix bottom': true,
#     'mass': [$(grep -i mass $file|awk '{printf "%5.2f,", $2}') ],
#     'timestep': $(grep -i timestep $file|awk '{printf "%3.1f:", $2}') ,
# },

def parse_cp2k_shell(folder, cp2k_shell_out, data_filter):

    cells = []
    symbols = []
    positions = []
    energies = []
    forces = []
    natoms = []

    data = {}

    for filename in cp2k_shell_out:

        with open(filename) as fin:
            lines = fin.readlines()

        nlines = len(lines)

        nconfigs = 0
        i = 0
        cell = None
        position = None
        energy = None
        force = None

        while (i<nlines):
            if "SET_CELL" in lines[i]:
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
                symbol = []
                for iatom in range(natom):
                    i += 1
                    l = lines[i].split()
                    pos_line = l[1:]
                    symbol += l[0]
                    position += [[float(x) for x in pos_line]]
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
                # dch = distance(cell, xyz[0], xyz[3])
                # dhcu = distance(cell, xyz[3], xyz[4:])
                # hcu = coord(dhcu, 0.1, 1.8, 9, 14)
                # dhcu = np.min(dhcu)

                # if (energy - -63894.772571900554) < 10:
                cells += [np.copy(cell).reshape([-1])]
                symbols += [np.copy(symbol)]
                positions += [np.copy(position).reshape([-1])]
                energies += [energy]
                forces += [np.copy(force).reshape([-1])]
                natoms += [natom]
                nconfigs += 1

                # print("dz", pos[5]-pos[8], np.max(np.abs(forces)))
                # else:
                #     logging.info(f"skip for large energy {energy}")
                # print(f"read config: {nconfigs} {energy}")
            i += 1

    if len(positions) > 0:
        data['positions'] = np.vstack(positions)
        data['forces'] = np.vstack(forces)
        data['energies'] = np.hstack(energies)
        data['cells'] = np.vstack(cells)
        data['symbols'] = np.vstack(symbols)
        data['nframes'] = len(positions)
        data['natoms'] = np.hstack(natoms)
        return data
    else:
        return {}

def parse_force_eval(folder, force_eval_pair, data_filter):

    cells = []
    symbols = []
    positions = []
    energies = []
    forces = []
    natoms = []
    nconfigs = 0

    data = {}

    sort_id = np.argsort([c[0] for c in force_eval_pair])

    all_files = []

    for filenames in np.array(force_eval_pair)[sort_id]:

        all_files += [i for i in filenames if i is not None]


        pos = read_pattern(filenames[1], {
            'table':r"\A\s?([A-Z][a-z]*?)\s+([+-]?\d+\.+\d+[eE]*[+-]*\d*)\s+([+-]?\d+\.+\d+[eE]*[+-]*\d*)\s+([+-]?\d+\.+\d+[eE]*[+-]*\d*)"})

        pos = np.array(pos['table'])
        if pos[0][0] == 'A':
            c = pos[:3]
            pos = pos[3:]
        else:
            c = pos[-3:]
            pos = pos[:-3]
        cell = np.array([[float(p[1]), float(p[2]), float(p[3])] for p in c], dtype=float)

        # pos = read_table_pattern(filenames[1],
        #                                         header_pattern=r"\&COORD",
        #                                         row_pattern=r"([a-zA-Z]+)\s+([+-]?\d+\.?\d?)\s+([+-]?\d+\.?\d?)\s+([+-]?\d+\.?\d?)",
        #                                         footer_pattern=r"\&END COORD",
        #                                         last_one_only=False
        #                                         )
        symbol = [p[0] for p in pos]
        position = np.array([[float(p[1]), float(p[2]), float(p[3])] for p in pos])

        # c = read_table_pattern(filenames[1],
        #                                         header_pattern=r"&CELL",
        #                                         row_pattern=r"([a-zA-Z]+)\s+([+-]?\d+\.*\d+)\s+([+-]?\d+\.*\d+)\s+([+-]?\d+\.*\d+)",
        #                                         footer_pattern=r"&END",
        #                                         last_one_only=False
        #                                         )
        # cell = np.array([[float(p[1]), float(p[2]), float(p[3])] for p in c])

        ffile = 2
        if filenames[2] is None:
            ffile = 0

        # fo = read_table_pattern(filenames[ffile],
        #                                        header_pattern=r"\sKind\s",
        #                                        row_pattern=r"([a-zA-Z]+)\s+([+-]?\d+\.*\d+)\s+([+-]?\d+\.*\d+)\s+([+-]?\d+\.*\d+)",
        #                                        footer_pattern=r"SUM OF ATOMIC",
        #                                        last_one_only=False
        #                                        )
        fo = read_pattern(filenames[ffile], {
            'table':r"\d+\s+\d+\s+([A-Z][a-z]*?)\s+([+-]?\d+\.+\d+[eE]*[+-]*\d*)\s+([+-]?\d+\.+\d+[eE]*[+-]*\d*)\s+([+-]?\d+\.+\d+[eE]*[+-]*\d*)"})

        natom = len(position)

        # fo = np.array(fo['table'][-natom:])
        fo = np.array(fo['table'])

        f_symbol = [p[0] for p in fo]
        force = np.array([[float(p[1]), float(p[2]), float(p[3])] for p in fo])*51.42208619083232
        logging.debug(repr(force))

        assert symbol == f_symbol, print(f"{repr(symbol)}\n{repr(f_symbol)}")

        d = read_pattern(filenames[0], {
            'converged':r"SCF run converged",
            'energy':r"Total FORCE_EVAL \( QS \) energy \(a.u.\):\s+([+-]?\d+\.*\d+)"})
        energy = float(d['energy'][0][0])*27.211396 # eV
        converged=False
        if len(d['converged']) > 0:
            converged=True
        if converged and len(positions) == len(forces):
            natoms += [len(positions)]
            cells += [np.copy(cell).reshape([-1])]
            symbols += [np.copy(symbol)]
            positions += [np.copy(position).reshape([-1])]
            energies += [energy]
            forces += [np.copy(force).reshape([-1])]
            nconfigs += 1

    if len(positions) > 0:
        data.update(parse_cp2k_inp(force_eval_pair[sort_id[-1]][1]))
        logging.info(repr(data))
        data['positions'] = np.vstack(positions)
        data['forces'] = np.vstack(forces)
        data['energies'] = np.hstack(energies)
        data['natoms'] = np.hstack(natoms)
        data['cells'] = np.vstack(cells)
        data['symbols'] = np.vstack(symbols)
        data['species'] = symbols[0]
        data['nframes'] = len(positions)
        data['filenames'] = np.hstack(all_files)

    return data
