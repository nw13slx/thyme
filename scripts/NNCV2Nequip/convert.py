import numpy as np
from thyme import *
from thyme.parsers.extxyz import write
import sys

if __name__ == "__main__":
    # trjs = from_file(sys.argv[1])
    # trjs.save(sys.argv[2]+".pickle")
    # trjs.save(sys.argv[2]+"_padded_mat.npz")
    trj = Trajectory.from_file(sys.argv[1], update_dict={
        CELL:[12.817769235175424, 24.094028158633765, 122.0],
        FIXED_ATTRS:[CELL],
        PER_FRAME_ATTRS: [POSITION, TOTAL_ENERGY, "label"],
        SPECIES: ["Au"]*144,
    }, mapping={
        POSITION:"xyz",
        TOTAL_ENERGY: "pe",
    })
    trj.include_frames(np.arange(0, len(trj), 100))
    print(trj)
    write(sys.argv[2], trj, append=False)
