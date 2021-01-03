import pytest
import numpy as np

from pytest import raises

from copy import deepcopy

from thyme.trajectory import Trajectory


@pytest.fixture(scope='class')
def dictionary():
    d = dict(
    positions=np.zeros([4, 5, 3]),
    symbols=np.array([['C']*5]*4),
    )
    yield d
    del d

class TestInit():

    def test_init(self):

        trj = Trajectory()

# class TestFromDict()
#
#     def test_from_

# def __init__(self)
# def __repr__(self)
# def __str__(self)
# def sanity_check(self)
# def add_per_frame_attr(name, item)
# def add_metadata(name, item)
# def filter_frames(self, accept_id=None)
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
# def skim(self, frame_list)
# def shuffle(self)
# def copy(self, otrj)
