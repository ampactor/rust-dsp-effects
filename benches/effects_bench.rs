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
use rust_dsp_effects::{Chorus, Delay, Distortion, Effect, LowPassFilter, Oversampled, WaveShape};

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
// Chorus Benchmarks
// ============================================================================

fn bench_chorus_single_sample(c: &mut Criterion) {
    let mut chorus = Chorus::new(SAMPLE_RATE);
    chorus.set_rate(2.0);
    chorus.set_depth(0.7);
    chorus.set_mix(0.5);
    chorus.reset(); // Snap params to avoid smoothing overhead

    c.bench_function("chorus/single_sample", |b| {
        b.iter(|| {
            let input = black_box(0.1);
            black_box(chorus.process(input))
        })
    });
}

fn bench_chorus_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("chorus/block");

    for size in BLOCK_SIZES {
        group.throughput(Throughput::Elements(*size as u64));

        let mut chorus = Chorus::new(SAMPLE_RATE);
        chorus.set_rate(2.0);
        chorus.set_depth(0.7);
        chorus.set_mix(0.5);
        chorus.reset();

        let input: Vec<f32> = (0..*size).map(|i| (i as f32 / 100.0).sin()).collect();
        let mut output = vec![0.0; *size];

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                chorus.process_block(black_box(&input), black_box(&mut output));
            })
        });
    }

    group.finish();
}

fn bench_chorus_parameter_smoothing(c: &mut Criterion) {
    let mut group = c.benchmark_group("chorus/parameter_smoothing");

    // Benchmark with vs without smoothing active
    let scenarios = [
        ("no_smoothing", false),
        ("with_smoothing", true),
    ];

    for (name, enable_smoothing) in &scenarios {
        let mut chorus = Chorus::new(SAMPLE_RATE);
        chorus.set_rate(2.0);
        chorus.set_depth(0.7);
        chorus.set_mix(0.5);

        if !enable_smoothing {
            chorus.reset(); // Snap to target, no smoothing
        }

        group.bench_function(*name, |b| {
            b.iter(|| {
                let input = black_box(0.1);
                black_box(chorus.process(input))
            })
        });
    }

    group.finish();
}

fn bench_chorus_rate_variations(c: &mut Criterion) {
    let mut group = c.benchmark_group("chorus/rate_variations");

    let rates = [
        ("slow_0.5hz", 0.5),
        ("medium_2hz", 2.0),
        ("fast_5hz", 5.0),
    ];

    for (name, rate) in &rates {
        let mut chorus = Chorus::new(SAMPLE_RATE);
        chorus.set_rate(*rate);
        chorus.set_depth(0.7);
        chorus.set_mix(0.5);
        chorus.reset();

        group.bench_function(*name, |b| {
            b.iter(|| {
                let input = black_box(0.1);
                black_box(chorus.process(input))
            })
        });
    }

    group.finish();
}

// ============================================================================
// Delay Benchmarks
// ============================================================================

fn bench_delay_single_sample(c: &mut Criterion) {
    let mut delay = Delay::new(SAMPLE_RATE);
    delay.set_delay_time_ms(375.0); // 1/8 note at 120 BPM
    delay.set_feedback(0.5);
    delay.set_mix(0.5);
    delay.reset();

    c.bench_function("delay/single_sample", |b| {
        b.iter(|| {
            let input = black_box(0.1);
            black_box(delay.process(input))
        })
    });
}

fn bench_delay_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("delay/block");

    for size in BLOCK_SIZES {
        group.throughput(Throughput::Elements(*size as u64));

        let mut delay = Delay::new(SAMPLE_RATE);
        delay.set_delay_time_ms(375.0);
        delay.set_feedback(0.5);
        delay.set_mix(0.5);
        delay.reset();

        let input: Vec<f32> = (0..*size).map(|i| (i as f32 / 100.0).sin()).collect();
        let mut output = vec![0.0; *size];

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                delay.process_block(black_box(&input), black_box(&mut output));
            })
        });
    }

    group.finish();
}

fn bench_delay_feedback_amounts(c: &mut Criterion) {
    let mut group = c.benchmark_group("delay/feedback_amounts");

    let feedback_levels = [
        ("no_feedback", 0.0),
        ("medium_feedback", 0.5),
        ("high_feedback", 0.9),
    ];

    for (name, feedback) in &feedback_levels {
        let mut delay = Delay::new(SAMPLE_RATE);
        delay.set_delay_time_ms(100.0);
        delay.set_feedback(*feedback);
        delay.set_mix(0.5);
        delay.reset();

        group.bench_function(*name, |b| {
            b.iter(|| {
                let input = black_box(0.1);
                black_box(delay.process(input))
            })
        });
    }

    group.finish();
}

fn bench_delay_times(c: &mut Criterion) {
    let mut group = c.benchmark_group("delay/delay_times");

    let times = [
        ("short_50ms", 50.0),
        ("medium_250ms", 250.0),
        ("long_1000ms", 1000.0),
    ];

    for (name, time_ms) in &times {
        let mut delay = Delay::new(SAMPLE_RATE);
        delay.set_delay_time_ms(*time_ms);
        delay.set_feedback(0.5);
        delay.set_mix(0.5);
        delay.reset();

        group.bench_function(*name, |b| {
            b.iter(|| {
                let input = black_box(0.1);
                black_box(delay.process(input))
            })
        });
    }

    group.finish();
}

// ============================================================================
// Filter Benchmarks
// ============================================================================

fn bench_filter_single_sample(c: &mut Criterion) {
    let mut filter = LowPassFilter::new(SAMPLE_RATE);
    filter.set_cutoff_hz(1000.0);
    filter.set_q(0.707);
    filter.reset();

    c.bench_function("filter/single_sample", |b| {
        b.iter(|| {
            let input = black_box(0.1);
            black_box(filter.process(input))
        })
    });
}

fn bench_filter_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter/block");

    for size in BLOCK_SIZES {
        group.throughput(Throughput::Elements(*size as u64));

        let mut filter = LowPassFilter::new(SAMPLE_RATE);
        filter.set_cutoff_hz(1000.0);
        filter.set_q(0.707);
        filter.reset();

        let input: Vec<f32> = (0..*size).map(|i| (i as f32 / 100.0).sin()).collect();
        let mut output = vec![0.0; *size];

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                filter.process_block(black_box(&input), black_box(&mut output));
            })
        });
    }

    group.finish();
}

fn bench_filter_cutoff_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter/cutoff_sweep");

    let cutoffs = [
        ("low_100hz", 100.0),
        ("mid_1khz", 1000.0),
        ("high_10khz", 10000.0),
    ];

    for (name, cutoff) in &cutoffs {
        let mut filter = LowPassFilter::new(SAMPLE_RATE);
        filter.set_cutoff_hz(*cutoff);
        filter.set_q(0.707);
        filter.reset();

        group.bench_function(*name, |b| {
            b.iter(|| {
                let input = black_box(0.1);
                black_box(filter.process(input))
            })
        });
    }

    group.finish();
}

fn bench_filter_resonance(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter/resonance");

    let q_values = [
        ("butterworth_0.707", 0.707),
        ("resonant_2.0", 2.0),
        ("high_res_10.0", 10.0),
    ];

    for (name, q) in &q_values {
        let mut filter = LowPassFilter::new(SAMPLE_RATE);
        filter.set_cutoff_hz(1000.0);
        filter.set_q(*q);
        filter.reset();

        group.bench_function(*name, |b| {
            b.iter(|| {
                let input = black_box(0.1);
                black_box(filter.process(input))
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

fn bench_chorus_creation(c: &mut Criterion) {
    c.bench_function("chorus/creation", |b| {
        b.iter(|| {
            let chorus = black_box(Chorus::new(SAMPLE_RATE));
            black_box(chorus)
        })
    });
}

fn bench_delay_creation(c: &mut Criterion) {
    c.bench_function("delay/creation", |b| {
        b.iter(|| {
            let delay = black_box(Delay::new(SAMPLE_RATE));
            black_box(delay)
        })
    });
}

fn bench_filter_creation(c: &mut Criterion) {
    c.bench_function("filter/creation", |b| {
        b.iter(|| {
            let filter = black_box(LowPassFilter::new(SAMPLE_RATE));
            black_box(filter)
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
    bench_chorus_single_sample,
    bench_chorus_block,
    bench_chorus_parameter_smoothing,
    bench_chorus_rate_variations,
    bench_chorus_creation,
    bench_delay_single_sample,
    bench_delay_block,
    bench_delay_feedback_amounts,
    bench_delay_times,
    bench_delay_creation,
    bench_filter_single_sample,
    bench_filter_block,
    bench_filter_cutoff_sweep,
    bench_filter_resonance,
    bench_filter_creation,
);

criterion_main!(benches);
