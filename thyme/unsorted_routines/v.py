
import logging
logging.basicConfig(filename=f'collect.log', filemode='w',
                    level=logging.INFO, format="%(message)s")
logging.getLogger().addHandler(logging.StreamHandler())
# from thyme.vasp.parser import pack_folder
    # folders = search_all_folders(['vasprun.xml', 'OUTCAR', 'vasp.out'])
# import thyme.file_op import search_all_folders
