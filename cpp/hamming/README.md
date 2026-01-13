# Hamming Distance Module

SIMD-optimized Hamming distance calculator for uint8 binary descriptors with zero-copy Python bindings.

## Features

- **SIMD optimizations**: AVX2 → SSE3/SSE2 → scalar fallback
- **Zero-copy**: Direct memory access to C-contiguous numpy arrays
- **Bit-level accuracy**: Counts differing bits, not bytes

## API

```python
import hamming

# Single distance
dist = hamming.hamming(a, b)  # a, b: (B,) uint8

# Query vs many descriptors
dists = hamming.hamming_many(query, descs)  # query: (B,), descs: (N,B) → (N,)

# Pairwise distances
dists = hamming.hamming_pairwise(a, b)  # a, b: (N,B) → (N,)

# Drop-in replacements (bit-level)
dist = hamming.hamming_distance(a, b)  # (B,) vs (B,) → int
dists = hamming.hamming_distances(a, b)  # (N,B) vs (N,B) → (N,)
```

## Requirements

- Input arrays must be `uint8` dtype
- Arrays must be C-contiguous (use `np.ascontiguousarray()` if needed)
- Shapes must match as specified
