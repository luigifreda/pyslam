import numpy as np
from numba import njit

import hamming


# ================================================================
# Hamming distance
# ================================================================

# Lookup table for popcount (number of set bits in a byte)
# Pre-computed for all 256 possible uint8 values
_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@njit(cache=True)
def _popcount_byte(x):
    """Count set bits in a single uint8 value using bit manipulation."""
    # Brian Kernighan's algorithm for counting set bits
    count = 0
    while x:
        count += 1
        x &= x - 1  # Clear the least significant set bit
    return count


@njit(cache=True)
def _popcount_array(x):
    """Count total set bits in uint8 array."""
    total = 0
    for i in range(len(x)):
        total += _popcount_byte(x[i])
    return total


def hamming_distance_numpy(a, b):
    """
    Compute bit-level Hamming distance between two uint8 descriptors using NumPy.

    Args:
        a: (B,) uint8 array
        b: (B,) uint8 array

    Returns:
        int: Number of differing bits
    """
    a = np.ascontiguousarray(a, dtype=np.uint8)
    b = np.ascontiguousarray(b, dtype=np.uint8)
    # XOR to find differing bits, then count set bits
    diff = np.bitwise_xor(a, b)
    return int(_popcount_array(diff))


def hamming_distances_numpy(a, b):
    """
    Compute bit-level Hamming distances using NumPy.

    Supports:
    - a: (B,), b: (N,B) -> returns (N,) distances from a to each row of b
    - a: (N,B), b: (N,B) -> returns (N,) pairwise distances

    Args:
        a: uint8 array, shape (B,) or (N,B)
        b: uint8 array, shape (N,B) or sequence of (B,) arrays

    Returns:
        (N,) uint16 array of distances
    """
    a = np.ascontiguousarray(a, dtype=np.uint8)

    # Handle case where b is a sequence/list
    if not isinstance(b, np.ndarray):
        # Convert sequence to array
        b = np.array([np.ascontiguousarray(bi, dtype=np.uint8) for bi in b])

    b = np.ascontiguousarray(b, dtype=np.uint8)

    if a.ndim == 1:
        # a: (B,), b: (N,B) -> compute distance from a to each row of b
        if b.ndim != 2 or b.shape[1] != a.shape[0]:
            raise ValueError(
                f"hamming_distances: if a is 1D (B,), b must be 2D (N,B). Got a.shape={a.shape}, b.shape={b.shape}"
            )
        diff = np.bitwise_xor(a[np.newaxis, :], b)  # (N,B)
        return np.array([_popcount_array(diff[i]) for i in range(len(diff))], dtype=np.uint16)
    elif a.ndim == 2:
        # a: (N,B), b: (N,B) -> pairwise distances
        if b.ndim != 2 or a.shape != b.shape:
            raise ValueError(
                f"hamming_distances: if a is 2D (N,B), b must be 2D (N,B) with matching shape. Got a.shape={a.shape}, b.shape={b.shape}"
            )
        diff = np.bitwise_xor(a, b)  # (N,B)
        return np.array([_popcount_array(diff[i]) for i in range(len(diff))], dtype=np.uint16)
    else:
        raise ValueError(f"hamming_distances: a must be 1D or 2D. Got a.ndim={a.ndim}")


# Current: using C++ module (fastest, SIMD-optimized)
# To use NumPy implementations instead, uncomment the following lines:
# hamming_distance = hamming_distance_numpy
# hamming_distances = hamming_distances_numpy

hamming_distance = hamming.hamming_distance
hamming_distances = hamming.hamming_distances


# ================================================================
# L2 distance
# ================================================================


@njit(cache=True)
def l2_distance(a, b):
    return np.linalg.norm(a.ravel() - b.ravel())


def l2_distances(a, b):
    return np.linalg.norm(a - b, axis=-1, keepdims=True)
