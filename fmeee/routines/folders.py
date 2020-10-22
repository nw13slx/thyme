import logging
import numpy as np

from glob import glob

from os import walk, mkdir
from os.path import isdir, isfile

from fmeee.trajectories import Trajectories

def parse_folders_trjs(folders, pack_folder_trj, data_filter, npz_filename=""):

    folders = sorted(folders)
    logging.info(f"all folders: {folders}")

    count = 0
    trjs = Trajectories()
    alldata = trjs.alldata
    for folder in folders:

        if folder == "./":
            casename = "current_folder"
        if folder[:2] == "./":
            casename = "_".join(folder[2:].split("/"))
        else:
            casename = "_".join(folder.split("/"))

        logging.info(casename)

        new_trj = pack_folder_trj(folder, data_filter)
        if new_trj.nframes >= 1:
            logging.info(f"save {folder} as {casename} : {new_trj.nframes} frames")
            new_trj.name = casename
            alldata[casename] = new_trj
            count += 1
            if count%10 == 0 and len(npz_filename)>0:
                trjs.save(f"{npz_filename}")
        else:
            logging.info(f"! skip whole folder {casename}, {new_trj.nframes}")

    if len(npz_filename)>0:
        trjs.save(f"{npz_filename}")
        logging.info(f"save as {npz_filename}")

    logging.info("Complete parsing")

    return trjs

def parse_folders(folders, pack_folder, data_filter, npz_filename):

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

        data = pack_folder(folder, data_filter)
        if data['nframes'] >= 1:
            logging.info(f"save {folder} as {casename} : {data['nframes']} frames")
            alldata[casename] = data
            count += 1
            if count%10 == 0:
                np.savez(f"{npz_filename}.npz", **alldata)
        else:
            logging.info(f"! skip whole folder {casename}, {data['nframes']}")

    np.savez(f"{npz_filename}.npz", **alldata)

    logging.info("Complete parsing")

    return alldata

def find_folders_matching(filenames, path):

    folders = []
    for root, dirs, files in walk("./"):
        for filename in filenames:
            if len(glob(f"{root}/{filename}")) > 0:
                folders += [root]
    return set(folders)

def find_folders(filenames, path):

    result = set([root \
                  for root, dirs, files in walk(path) \
                  if len((set(files)).intersection(filenames)) >0])
    return result

def safe_mkdir(name):
    if not isdir(name):
        mkdir(name)
