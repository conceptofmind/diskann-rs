//! Quantized search example — demonstrates building and searching
//! with PQ, F16, and Int8 quantization fused into graph traversal.
//!
//! Run with: cargo run --example quantized_search --release

use diskann_rs::pq::PQConfig;
use diskann_rs::{DiskAnnParams, DistL2, QuantizedConfig, QuantizedDiskANN};
use rand::prelude::*;
use rand::SeedableRng;
use std::collections::HashSet;
use std::time::Instant;

fn main() {
    let dim = 64;
    let n_vectors = 2_000;
    let n_queries = 50;
    let k = 10;
    let beam_width = 64;

    println!("Quantized DiskANN Search Example");
    println!("================================");
    println!(
        "Vectors: {}, Dim: {}, Queries: {}, k: {}, Beam: {}\n",
        n_vectors, dim, n_queries, k, beam_width
    );

    // Generate random vectors
    let mut rng = StdRng::seed_from_u64(42);
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
        .collect();

    // Ground truth (brute force)
    let ground_truth: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(u32, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
                    (i as u32, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.iter().take(k).map(|(i, _)| *i).collect()
        })
        .collect();

    let ann_params = DiskAnnParams {
        max_degree: 32,
        build_beam_width: 128,
        alpha: 1.2,
    };

    println!(
        "| {:<12} | {:<10} | {:<12} | {:<12} | {:<10} |",
        "Method", "Rerank", "Build (ms)", "Search (ms)", "Recall@10"
    );
    println!("|{:-<14}|{:-<12}|{:-<14}|{:-<14}|{:-<12}|", "", "", "", "", "");

    // --- F16 ---
    {
        let path = "example_quantized_f16.db";
        let _ = std::fs::remove_file(path);

        let start = Instant::now();
        let index = QuantizedDiskANN::<DistL2>::build_f16(
            &vectors,
            DistL2 {},
            path,
            ann_params,
            QuantizedConfig { rerank_size: 0 },
        )
        .unwrap();
        let build_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        let recall = avg_recall(&index, &queries, &ground_truth, k, beam_width);
        let search_ms = start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "| {:<12} | {:<10} | {:<12.1} | {:<12.1} | {:>9.1}% |",
            "F16",
            "none",
            build_ms,
            search_ms,
            recall * 100.0
        );
        let _ = std::fs::remove_file(path);
    }

    // --- Int8 ---
    {
        let path = "example_quantized_int8.db";
        let _ = std::fs::remove_file(path);

        let start = Instant::now();
        let index = QuantizedDiskANN::<DistL2>::build_int8(
            &vectors,
            DistL2 {},
            path,
            ann_params,
            QuantizedConfig { rerank_size: 0 },
        )
        .unwrap();
        let build_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        let recall = avg_recall(&index, &queries, &ground_truth, k, beam_width);
        let search_ms = start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "| {:<12} | {:<10} | {:<12.1} | {:<12.1} | {:>9.1}% |",
            "Int8",
            "none",
            build_ms,
            search_ms,
            recall * 100.0
        );
        let _ = std::fs::remove_file(path);
    }

    // --- PQ ---
    {
        let path = "example_quantized_pq.db";
        let _ = std::fs::remove_file(path);

        let pq_config = PQConfig {
            num_subspaces: 8,
            num_centroids: 256,
            kmeans_iterations: 15,
            training_sample_size: 0,
        };

        let start = Instant::now();
        let index = QuantizedDiskANN::<DistL2>::build_pq(
            &vectors,
            DistL2 {},
            path,
            ann_params,
            pq_config,
            QuantizedConfig { rerank_size: 0 },
        )
        .unwrap();
        let build_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        let recall = avg_recall(&index, &queries, &ground_truth, k, beam_width);
        let search_ms = start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "| {:<12} | {:<10} | {:<12.1} | {:<12.1} | {:>9.1}% |",
            "PQ-8",
            "none",
            build_ms,
            search_ms,
            recall * 100.0
        );
        let _ = std::fs::remove_file(path);
    }

    // --- PQ + rerank ---
    {
        let path = "example_quantized_pq_rr.db";
        let _ = std::fs::remove_file(path);

        let pq_config = PQConfig {
            num_subspaces: 8,
            num_centroids: 256,
            kmeans_iterations: 15,
            training_sample_size: 0,
        };

        let start = Instant::now();
        let index = QuantizedDiskANN::<DistL2>::build_pq(
            &vectors,
            DistL2 {},
            path,
            ann_params,
            pq_config,
            QuantizedConfig { rerank_size: 50 },
        )
        .unwrap();
        let build_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        let recall = avg_recall(&index, &queries, &ground_truth, k, beam_width);
        let search_ms = start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "| {:<12} | {:<10} | {:<12.1} | {:<12.1} | {:>9.1}% |",
            "PQ-8",
            "top-50",
            build_ms,
            search_ms,
            recall * 100.0
        );
        let _ = std::fs::remove_file(path);
    }

    println!("\nDone.");
}

fn avg_recall(
    index: &QuantizedDiskANN<DistL2>,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<u32>],
    k: usize,
    beam_width: usize,
) -> f32 {
    let mut total = 0.0f32;
    for (query, gt) in queries.iter().zip(ground_truth) {
        let results = index.search(query, k, beam_width);
        let gt_set: HashSet<u32> = gt.iter().copied().collect();
        let hits = results.iter().filter(|id| gt_set.contains(id)).count();
        total += hits as f32 / k as f32;
    }
    total / queries.len() as f32
}
