import logging
logging.basicConfig(filename=f'collect.log', filemode='w',
                                          level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

import numpy as np
from ase.atoms import Atoms

from fmeee.routines.folders import parse_folders
from fmeee.parsers.tfmd import pack_folder, get_childfolders

def main():

    folders = get_childfolders("./", include_xyz = True)
    parse_folders(folders, pack_folder, e_filter, "all_data")

def e_filter(xyz, f, e, c, species):
    return True

if __name__ == '__main__':
    main()
