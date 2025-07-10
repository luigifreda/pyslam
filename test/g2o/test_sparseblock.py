import sys 
import numpy as np
import math 

sys.path.append("../../")
from pyslam.config import Config

import g2o


def print_sparse_block_matrix(mat):
    num_block_rows = len(mat.row_block_indices)  # number of block rows
    num_block_cols = len(mat.col_block_indices)  # number of block cols

    for r in range(num_block_rows):
        for c in range(num_block_cols):
            block = mat.block(r, c)
            if block.size != 0:
                print(f"Block ({r}, {c}) [{block.shape[0]}x{block.shape[1]}]:")
                print(block)
            else:
                print(f"Block ({r}, {c}): [empty]")
                

if __name__ == "__main__":

    rbi = [0, 3, 6]
    cbi = [0, 3, 6]
    mat = g2o.SparseBlockMatrixX(rbi, cbi, 2, 2)

    blk00 = mat.block_ptr(0, 0, True)
    blk00[:] = np.eye(blk00.shape[0], blk00.shape[1])

    blk11 = mat.block_ptr(1, 1, True)
    blk11[:] = np.ones((blk11.shape[0], blk11.shape[1]))

    print("Matrix size:", mat.rows(), mat.cols())
    print_sparse_block_matrix(mat)