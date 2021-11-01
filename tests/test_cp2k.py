import pytest
import numpy as np

from pytest import raises

from copy import deepcopy

from thyme.parsers.cp2k import parse_md, pack_folder_trj
from thyme.trajectory import Trajectory

from pathlib import Path


class TestParseMD:
    def test_direct(self):
        folder = "tests/example_files/cp2k_md"
        trj = parse_md(folder, f"{folder}/CP2K.inp", "CO2H2")

        # check whether all frames are read
        assert len(trj) == 3

        # check whether it parsed the fixed atom and nan their entries correctly
        assert (trj.force[:, 0, :] != trj.force[:, 0, :]).all()
        assert (trj.force[:, 1:, :] == trj.force[:, 1:, :]).all()

    def test_foldersearch(self):
        folder = "tests/example_files/cp2k_md"
        trj = pack_folder_trj(folder=folder, data_filter=None)
        assert len(trj) == 3


class TestASEShellOut:
    def test_direct(self):
        pass
