import pytest
import numpy as np

from pytest import raises

from copy import deepcopy

from thyme.trajectory import Trajectory


def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []
    for scenario in metafunc.cls.scenarios:
        idlist.append(scenario[0])
        items = scenario[1].items()
        argnames = [x[0] for x in items]
        argvalues.append([x[1] for x in items])
    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


scenario1 = ("empty", {"trj": Trajectory()})
scenario2 = ("prefilled", {"trj": None})


class TestEmpty:

    scenarios = [scenario1, scenario2]
    trj = None

    def test_init(self, trj):
        self.trj = "aaa"
        if trj is None:
            self.trj = "1"
            print("hello", self.trj)
        pass

    def test_repr(self, trj):
        print(repr(trj))

    def test_str(self, trj):
        print(trj)
        print("hello", self.trj)
        self.trj = "2"

    def test_sanity_check(self, trj):
        print("hello", self.trj)
        self.trj = "3"
        pass
        # trj.sanity_check()

    def test_add_p_attr(self, trj):
        print("hello", self.trj)
        pass
        # trj.add_per_frame_attr("hello", [])

    def test_add_m_attr(self, trj):
        pass
        # trj.add_per_frame_attr("hello", [])


# def sanity_check(self)
# def add_per_frame_attr(name, item)
# def add_metadata(name, item)
# def include_frames(self, accept_id=None)
# def from_file(filename)
# def from_dict(dictionary)
# def to_dict(self)
# def copy_dict(self, dictionary)
# def copy_metadata(self, trj, exception)
# def save(self, name str, format str = None)
# def clean_containers(self)
# def add_containers(self, natom int = 0,
# def append_frame_from_dict(self, dictionary dict, nframes int,
# def add_frames_from_dict(self, dictionary dict, nframes int,
# def from_padded_trajectory(otrj)
# def add_trj(self, trj)
# def convert_to_np(self)
# def convert_to_list(self)
# def reshape(self)
# def flatten(self)
# def extract_frames(self, frame_list)
# def shuffle(self)
# def copy(self, otrj)
