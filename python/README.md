# pydiskann

Python bindings for [diskann-rs](https://github.com/lukaesch/diskann-rs): memory-mapped
Vamana/DiskANN vector search with filtered search, incremental updates and quantization
(F16, Int8, PQ, RaBitQ).

```bash
uv add pydiskann
```

```python
import pydiskann as d

idx = d.DiskANN.build(vectors, "index.db", metric="l2")
idx.search(query, k=10, beam_width=128)
```

Vectors are `list[list[float]]` (`arr.tolist()` for numpy). `l2` returns squared distances.


# Build
```py
uvx --with ziglang maturin build --release --zig --compatibility manylinux_2_17 --out dist --sdist
```