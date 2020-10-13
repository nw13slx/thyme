
class Trajectory():

    default_keys = ["positions", "forces", "energies",
                    "cells", "nelm", "nframes", "cutoff",
                    "dipole_correction", "species", "natom", "kpoints"]

    def __init__(self):

        positions = None
        forces = None
        energies = None
        cells = None
        nelm = None
        nframes = 0
        cutoff = None
        dipole_correction = None
        species = None
        natom = None
        kpoints = None
