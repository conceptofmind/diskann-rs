# DiskANN Implementation in Rust

A fork of https://docs.rs/diskann_rs/latest/diskann_rs/

Tests:
```
cargo clean &&
CC=gcc-12 cargo build --release &&
CC=gcc-12 cargo test --release
```
```
CC=gcc-12 uv run pytest tests/tests.py
```