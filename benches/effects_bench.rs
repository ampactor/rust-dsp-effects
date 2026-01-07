//! Benchmarks for audio effects.
//!
//! Run with: cargo bench
//!
//! These benchmarks measure:
//! - Single sample processing latency
//! - Block processing throughput
//! - Oversampling overhead
//! - Parameter smoothing cost

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rust_dsp_effects::{Distortion, Effect, Oversampled, WaveShape};

const SAMPLE_RATE: f32 = 48000.0;
const BLOCK_SIZES: &[usize] = &[64, 128, 256, 512, 1024];

// ============================================================================
// Distortion Benchmarks
// ============================================================================

fn bench_distortion_single_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("distortion/single_sample");

    // Benchmark each waveshaping algorithm
    let waveshapes = [
        ("soft_clip", WaveShape::SoftClip),
        ("hard_clip", WaveShape::HardClip),
        ("foldback", WaveShape::Foldback),
        ("asymmetric", WaveShape::Asymmetric),
    ];

    for (name, waveshape) in &waveshapes {
        let mut dist = Distortion::new(SAMPLE_RATE);
        dist.set_waveshape(*waveshape);
        dist.set_drive_db(20.0);
        dist.set_tone_hz(4000.0);
        dist.reset(); // Snap params to avoid smoothing overhead

        group.bench_function(*name, |b| {
            b.iter(|| {
                let input = black_box(0.1);
                black_box(dist.process(input))
            })
        });
    }

    group.finish();
}

fn bench_distortion_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("distortion/block");

    for size in BLOCK_SIZES {
        group.throughput(Throughput::Elements(*size as u64));

        let mut dist = Distortion::new(SAMPLE_RATE);
        dist.set_drive_db(20.0);
        dist.set_tone_hz(4000.0);
        dist.reset();

        let input: Vec<f32> = (0..*size).map(|i| (i as f32 / 100.0).sin()).collect();
        let mut output = vec![0.0; *size];

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                dist.process_block(black_box(&input), black_box(&mut output));
            })
        });
    }

    group.finish();
}

fn bench_distortion_oversampled(c: &mut Criterion) {
    let mut group = c.benchmark_group("distortion/oversampled");

    for size in BLOCK_SIZES {
        group.throughput(Throughput::Elements(*size as u64));

        let input: Vec<f32> = (0..*size).map(|i| (i as f32 / 100.0).sin()).collect();
        let mut output = vec![0.0; *size];

        // 1x (no oversampling) - just plain distortion
        {
            let mut dist = Distortion::new(SAMPLE_RATE);
            dist.set_drive_db(20.0);
            dist.set_tone_hz(4000.0);
            dist.reset();

            group.bench_with_input(BenchmarkId::new("1x", size), size, |b, _| {
                b.iter(|| {
                    dist.process_block(black_box(&input), black_box(&mut output));
                })
            });
        }

        // 2x oversampling
        {
            let dist = Distortion::new(SAMPLE_RATE);
            let mut oversampled = Oversampled::<2, _>::new(dist, SAMPLE_RATE);
            oversampled.inner_mut().set_drive_db(20.0);
            oversampled.inner_mut().set_tone_hz(4000.0);
            oversampled.inner_mut().reset();

            group.bench_with_input(BenchmarkId::new("2x", size), size, |b, _| {
                b.iter(|| {
                    oversampled.process_block(black_box(&input), black_box(&mut output));
                })
            });
        }

        // 4x oversampling
        {
            let dist = Distortion::new(SAMPLE_RATE);
            let mut oversampled = Oversampled::<4, _>::new(dist, SAMPLE_RATE);
            oversampled.inner_mut().set_drive_db(20.0);
            oversampled.inner_mut().set_tone_hz(4000.0);
            oversampled.inner_mut().reset();

            group.bench_with_input(BenchmarkId::new("4x", size), size, |b, _| {
                b.iter(|| {
                    oversampled.process_block(black_box(&input), black_box(&mut output));
                })
            });
        }
    }

    group.finish();
}

fn bench_distortion_parameter_smoothing(c: &mut Criterion) {
    let mut group = c.benchmark_group("distortion/parameter_smoothing");

    // Benchmark with vs without smoothing active
    let scenarios = [
        ("no_smoothing", false),
        ("with_smoothing", true),
    ];

    for (name, enable_smoothing) in &scenarios {
        let mut dist = Distortion::new(SAMPLE_RATE);
        dist.set_drive_db(20.0);
        dist.set_tone_hz(4000.0);

        if !enable_smoothing {
            dist.reset(); // Snap to target, no smoothing
        }
        // If enable_smoothing=true, params will smooth from their initial values

        group.bench_function(*name, |b| {
            b.iter(|| {
                let input = black_box(0.1);
                black_box(dist.process(input))
            })
        });
    }

    group.finish();
}

// ============================================================================
// Memory/Allocation Benchmarks
// ============================================================================

fn bench_distortion_creation(c: &mut Criterion) {
    c.bench_function("distortion/creation", |b| {
        b.iter(|| {
            let dist = black_box(Distortion::new(SAMPLE_RATE));
            black_box(dist)
        })
    });
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    benches,
    bench_distortion_single_sample,
    bench_distortion_block,
    bench_distortion_oversampled,
    bench_distortion_parameter_smoothing,
    bench_distortion_creation,
);

criterion_main!(benches);
