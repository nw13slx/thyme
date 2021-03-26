from thyme.parsers.flare import from_file
import sys

if __name__ == "__main__":
    trjs = from_file(sys.argv[1])
    trjs.save(sys.argv[2]+".pickle")
    trjs.save(sys.argv[2]+"_padded_mat.npz")
