from thyme.parsers.flare import from_file
import sys

if __name__ == "__main__":
    # trjs = from_file(sys.argv[1])
    # trjs.save(sys.argv[2]+".pickle")
    # trjs.save(sys.argv[2]+"_padded_mat.npz")
    trjs = Trajectories.from_file("all.pickle")
    trj = list(trjs.all_data.values())[0]
    from thyme.routines.manipulate import replicate
    trj = replicate(trj, (2, 2, 2))
    trj.save(sys.argv[2]+"re_padded_mat.npz")

