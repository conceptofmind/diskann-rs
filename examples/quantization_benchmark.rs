//! Quantization comparison benchmark
//!
//! Run with: cargo run --example quantization_benchmark --release
//!
//! Outputs data for compression ratio vs recall chart.

use diskann_rs::pq::{ProductQuantizer, PQConfig};
use diskann_rs::sq::{F16Quantizer, Int8Quantizer, VectorQuantizer};
use rand::prelude::*;
use rand::SeedableRng;
use std::time::Instant;

fn main() {
    let dim = 128;
    let n_vectors = 10_000;
    let n_queries = 100;
    let k = 10;

    println!("Quantization Benchmark");
    println!("======================");
    println!("Vectors: {}, Dim: {}, Queries: {}, k: {}\n", n_vectors, dim, n_queries, k);

    // Generate random vectors
    let mut rng = StdRng::seed_from_u64(42);
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect())
        .collect();

    // Compute ground truth (exact k-NN)
    let ground_truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, l2_squared(q, v)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.iter().take(k).map(|(i, _)| *i).collect()
        })
        .collect();

    println!("| Method | Compression | Code Size | Encode Time | Search Time | Recall@{} |", k);
    println!("|--------|-------------|-----------|-------------|-------------|----------|");

    // Baseline (no compression)
    let baseline_size = dim * 4;
    println!(
        "| None (f32) | 1.0x | {} B | - | - | 100.0% |",
        baseline_size
    );

    // F16 Quantizer
    {
        let q = F16Quantizer::new(dim);
        let code_size = dim * 2;
        let compression = baseline_size as f32 / code_size as f32;

        let start = Instant::now();
        let codes: Vec<Vec<u8>> = vectors.iter().map(|v| q.encode(v)).collect();
        let encode_time = start.elapsed();

        let start = Instant::now();
        let recall = compute_recall(&q, &queries, &codes, &ground_truth, k);
        let search_time = start.elapsed();

        println!(
            "| F16 | {:.1}x | {} B | {:.1}ms | {:.1}ms | {:.1}% |",
            compression,
            code_size,
            encode_time.as_secs_f64() * 1000.0,
            search_time.as_secs_f64() * 1000.0,
            recall * 100.0
        );
    }

    // Int8 Quantizer
    {
        let q = Int8Quantizer::train(&vectors).unwrap();
        let code_size = dim;
        let compression = baseline_size as f32 / code_size as f32;

        let start = Instant::now();
        let codes: Vec<Vec<u8>> = vectors.iter().map(|v| q.encode(v)).collect();
        let encode_time = start.elapsed();

        let start = Instant::now();
        let recall = compute_recall(&q, &queries, &codes, &ground_truth, k);
        let search_time = start.elapsed();

        println!(
            "| Int8 | {:.1}x | {} B | {:.1}ms | {:.1}ms | {:.1}% |",
            compression,
            code_size,
            encode_time.as_secs_f64() * 1000.0,
            search_time.as_secs_f64() * 1000.0,
            recall * 100.0
        );
    }

    // PQ with different subspace counts
    for num_subspaces in [8, 16, 32] {
        let config = PQConfig {
            num_subspaces,
            num_centroids: 256,
            kmeans_iterations: 15,
            training_sample_size: 5000,
        };
        let q = ProductQuantizer::train(&vectors, config).unwrap();
        let code_size = num_subspaces;
        let compression = baseline_size as f32 / code_size as f32;

        let start = Instant::now();
        let codes: Vec<Vec<u8>> = vectors.iter().map(|v| q.encode(v)).collect();
        let encode_time = start.elapsed();

        let start = Instant::now();
        let recall = compute_recall_pq(&q, &queries, &codes, &ground_truth, k);
        let search_time = start.elapsed();

        println!(
            "| PQ-{} | {:.1}x | {} B | {:.1}ms | {:.1}ms | {:.1}% |",
            num_subspaces,
            compression,
            code_size,
            encode_time.as_secs_f64() * 1000.0,
            search_time.as_secs_f64() * 1000.0,
            recall * 100.0
        );
    }

    println!("\n# Chart Data (CSV)");
    println!("method,compression,recall");
    println!("None,1.0,100.0");

    // Regenerate for CSV output
    {
        let q = F16Quantizer::new(dim);
        let codes: Vec<Vec<u8>> = vectors.iter().map(|v| q.encode(v)).collect();
        let recall = compute_recall(&q, &queries, &codes, &ground_truth, k);
        println!("F16,2.0,{:.1}", recall * 100.0);
    }
    {
        let q = Int8Quantizer::train(&vectors).unwrap();
        let codes: Vec<Vec<u8>> = vectors.iter().map(|v| q.encode(v)).collect();
        let recall = compute_recall(&q, &queries, &codes, &ground_truth, k);
        println!("Int8,4.0,{:.1}", recall * 100.0);
    }
    for num_subspaces in [32, 16, 8] {
        let config = PQConfig {
            num_subspaces,
            num_centroids: 256,
            kmeans_iterations: 15,
            training_sample_size: 5000,
        };
        let q = ProductQuantizer::train(&vectors, config).unwrap();
        let codes: Vec<Vec<u8>> = vectors.iter().map(|v| q.encode(v)).collect();
        let recall = compute_recall_pq(&q, &queries, &codes, &ground_truth, k);
        let compression = baseline_size as f32 / num_subspaces as f32;
        println!("PQ-{},{:.1},{:.1}", num_subspaces, compression, recall * 100.0);
    }
}

fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn compute_recall(
    q: &dyn VectorQuantizer,
    queries: &[Vec<f32>],
    codes: &[Vec<u8>],
    ground_truth: &[Vec<usize>],
    k: usize,
) -> f32 {
    let mut total_recall = 0.0;
    for (query, gt) in queries.iter().zip(ground_truth) {
        let mut dists: Vec<(usize, f32)> = codes
            .iter()
            .enumerate()
            .map(|(i, c)| (i, q.asymmetric_distance(query, c)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let retrieved: std::collections::HashSet<usize> =
            dists.iter().take(k).map(|(i, _)| *i).collect();
        let gt_set: std::collections::HashSet<usize> = gt.iter().copied().collect();
        let hits = retrieved.intersection(&gt_set).count();
        total_recall += hits as f32 / k as f32;
    }
    total_recall / queries.len() as f32
}

fn compute_recall_pq(
    q: &ProductQuantizer,
    queries: &[Vec<f32>],
    codes: &[Vec<u8>],
    ground_truth: &[Vec<usize>],
    k: usize,
) -> f32 {
    let mut total_recall = 0.0;
    for (query, gt) in queries.iter().zip(ground_truth) {
        let table = q.create_distance_table(query);
        let mut dists: Vec<(usize, f32)> = codes
            .iter()
            .enumerate()
            .map(|(i, c)| (i, q.distance_with_table(&table, c)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let retrieved: std::collections::HashSet<usize> =
            dists.iter().take(k).map(|(i, _)| *i).collect();
        let gt_set: std::collections::HashSet<usize> = gt.iter().copied().collect();
        let hits = retrieved.intersection(&gt_set).count();
        total_recall += hits as f32 / k as f32;
    }
    total_recall / queries.len() as f32
}
