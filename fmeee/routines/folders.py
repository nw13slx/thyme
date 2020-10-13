import logging
import numpy as np

from os import walk, mkdir
from os.path import isdir, isfile

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
            logging.info(f"{folder}, {casename}, {data['nframes']}")
            alldata[casename] = data
            count += 1
            if count%10 == 0:
                np.savez(f"{npz_filename}.npz", **alldata)
        else:
            logging.info(f"! skip whole folder {casename}, {data['nframes']}")

    np.savez(f"{npz_filename}.npz", **alldata)

    logging.info("Complete parsing")

    return alldata


def find_folders(filenames, path):

    result = set([root \
                  for root, dirs, files in walk(path) \
                  if len((set(files)).intersection(filenames)) >0])
    return result

def safe_mkdir(name):
    if not isdir(name):
        mkdir(name)