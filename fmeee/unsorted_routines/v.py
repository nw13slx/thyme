
import logging
logging.basicConfig(filename=f'collect.log', filemode='w',
                    level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())
# from fmeee.vasp.parser import pack_folder
    # folders = search_all_folders(['vasprun.xml', 'OUTCAR', 'vasp.out'])
# import fmeee.file_op import search_all_folders
