//! Dynamics compressor with envelope follower and gain reduction.
//!
//! Provides classic dynamics compression for controlling signal levels,
//! evening out performances, and adding punch to audio.

use crate::effect::Effect;
use crate::param::SmoothedParam;

/// Converts linear amplitude to decibels.
#[inline]
fn linear_to_db(linear: f32) -> f32 {
    20.0 * linear.max(1e-6).log10()
}

/// Converts decibels to linear amplitude.
#[inline]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Envelope follower for tracking signal amplitude.
///
/// Uses peak detection with separate attack and release times
/// for natural-sounding dynamics response.
#[derive(Debug, Clone)]
struct EnvelopeFollower {
    /// Current envelope level (linear)
    envelope: f32,
    /// Attack coefficient
    attack_coeff: f32,
    /// Release coefficient
    release_coeff: f32,
    /// Sample rate
    sample_rate: f32,
}

impl EnvelopeFollower {
    fn new(sample_rate: f32) -> Self {
        let mut follower = Self {
            envelope: 0.0,
            attack_coeff: 0.0,
            release_coeff: 0.0,
            sample_rate,
        };
        follower.set_attack_ms(10.0);
        follower.set_release_ms(100.0);
        follower
    }

    /// Sets the attack time in milliseconds.
    fn set_attack_ms(&mut self, attack_ms: f32) {
        // Time constant for exponential smoothing
        // coeff = exp(-1 / (time_ms * sample_rate / 1000))
        self.attack_coeff = (-1.0 / (attack_ms * self.sample_rate / 1000.0)).exp();
    }

    /// Sets the release time in milliseconds.
    fn set_release_ms(&mut self, release_ms: f32) {
        self.release_coeff = (-1.0 / (release_ms * self.sample_rate / 1000.0)).exp();
    }

    /// Processes a sample and returns the envelope level.
    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let input_abs = input.abs();

        // Choose attack or release based on whether signal is rising or falling
        let coeff = if input_abs > self.envelope {
            self.attack_coeff
        } else {
            self.release_coeff
        };

        // Exponential smoothing: y[n] = coeff * y[n-1] + (1 - coeff) * x[n]
        self.envelope = coeff * self.envelope + (1.0 - coeff) * input_abs;
        self.envelope
    }

    fn reset(&mut self) {
        self.envelope = 0.0;
    }
}

/// Gain computer for calculating compression curve.
///
/// Implements threshold, ratio, and soft knee characteristics.
#[derive(Debug, Clone)]
struct GainComputer {
    threshold_db: f32,
    ratio: f32,
    knee_db: f32,
}

impl GainComputer {
    fn new() -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 4.0,
            knee_db: 6.0,
        }
    }

    /// Calculates gain reduction in dB for given input level.
    ///
    /// Uses soft knee for smooth transition into compression.
    #[inline]
    fn compute_gain_db(&self, input_db: f32) -> f32 {
        let overshoot = input_db - self.threshold_db;

        // Soft knee implementation
        if overshoot <= -self.knee_db / 2.0 {
            // Below knee - no compression
            0.0
        } else if overshoot > self.knee_db / 2.0 {
            // Above knee - full compression
            let gain_reduction = overshoot * (1.0 - 1.0 / self.ratio);
            -gain_reduction
        } else {
            // Inside knee - smooth transition
            let knee_factor = (overshoot + self.knee_db / 2.0) / self.knee_db;
            let gain_reduction = knee_factor * knee_factor * overshoot * (1.0 - 1.0 / self.ratio);
            -gain_reduction
        }
    }
}

/// Dynamics compressor effect.
///
/// Reduces the dynamic range of audio by attenuating loud signals above
/// a threshold while leaving quieter signals unchanged.
///
/// # Parameters
///
/// - **Threshold** (-60 to 0 dB): Level where compression starts
/// - **Ratio** (1:1 to 20:1): Amount of compression (higher = more)
/// - **Attack** (0.1 to 100 ms): How quickly compression engages
/// - **Release** (10 to 1000 ms): How quickly compression disengages
/// - **Knee** (0 to 12 dB): Soft (>0) vs hard (0) knee
/// - **Makeup Gain** (0 to 24 dB): Output level boost
///
/// # Example
///
/// ```
/// use rust_dsp_effects::{Compressor, Effect};
///
/// let mut comp = Compressor::new(44100.0);
/// comp.set_threshold_db(-20.0);  // Compress above -20dB
/// comp.set_ratio(4.0);            // 4:1 ratio
/// comp.set_attack_ms(5.0);        // Fast attack
/// comp.set_release_ms(50.0);      // Medium release
///
/// let output = comp.process(input);
/// ```
#[derive(Debug, Clone)]
pub struct Compressor {
    envelope_follower: EnvelopeFollower,
    gain_computer: GainComputer,

    /// Smoothed makeup gain (linear)
    makeup_gain: SmoothedParam,

    /// Sample rate
    sample_rate: f32,
}

impl Compressor {
    /// Creates a new compressor with default settings.
    ///
    /// # Default Parameters
    ///
    /// - Threshold: -20 dB
    /// - Ratio: 4:1
    /// - Attack: 10 ms
    /// - Release: 100 ms
    /// - Knee: 6 dB (soft)
    /// - Makeup gain: 0 dB
    pub fn new(sample_rate: f32) -> Self {
        Self {
            envelope_follower: EnvelopeFollower::new(sample_rate),
            gain_computer: GainComputer::new(),
            makeup_gain: SmoothedParam::with_config(1.0, sample_rate, 10.0), // 1.0 = 0dB
            sample_rate,
        }
    }

    /// Sets the threshold in dB.
    ///
    /// Signals above this level will be compressed.
    /// Typical values: -20 to -10 dB for general use
    pub fn set_threshold_db(&mut self, threshold_db: f32) {
        self.gain_computer.threshold_db = threshold_db.clamp(-60.0, 0.0);
    }

    /// Sets the compression ratio.
    ///
    /// - 1.0: No compression (1:1)
    /// - 2.0: Gentle compression (2:1)
    /// - 4.0: Moderate compression (4:1)
    /// - 10.0: Heavy compression (10:1)
    /// - 20.0: Near-limiting (20:1)
    pub fn set_ratio(&mut self, ratio: f32) {
        self.gain_computer.ratio = ratio.clamp(1.0, 20.0);
    }

    /// Sets the attack time in milliseconds.
    ///
    /// How quickly the compressor responds to peaks.
    /// - Fast (< 5ms): Catch transients, can sound pumpy
    /// - Medium (5-20ms): General purpose
    /// - Slow (> 20ms): Preserve transients, natural sound
    pub fn set_attack_ms(&mut self, attack_ms: f32) {
        self.envelope_follower.set_attack_ms(attack_ms.clamp(0.1, 100.0));
    }

    /// Sets the release time in milliseconds.
    ///
    /// How quickly the compressor stops compressing.
    /// - Fast (< 50ms): Pumping effect
    /// - Medium (50-200ms): General purpose
    /// - Slow (> 200ms): Smooth, transparent
    pub fn set_release_ms(&mut self, release_ms: f32) {
        self.envelope_follower.set_release_ms(release_ms.clamp(10.0, 1000.0));
    }

    /// Sets the knee width in dB.
    ///
    /// - 0 dB: Hard knee (abrupt compression)
    /// - 6 dB: Soft knee (smooth compression)
    /// - 12 dB: Very soft knee (gentle compression)
    pub fn set_knee_db(&mut self, knee_db: f32) {
        self.gain_computer.knee_db = knee_db.clamp(0.0, 12.0);
    }

    /// Sets the makeup gain in dB.
    ///
    /// Boosts the output to compensate for gain reduction.
    /// Typical: Set to roughly match the average gain reduction.
    pub fn set_makeup_gain_db(&mut self, gain_db: f32) {
        let linear = db_to_linear(gain_db.clamp(0.0, 24.0));
        self.makeup_gain.set_target(linear);
    }

    /// Resets the compressor state.
    pub fn reset(&mut self) {
        self.envelope_follower.reset();
        let makeup_target = self.makeup_gain.target();
        self.makeup_gain.set_immediate(makeup_target);
    }
}

impl Effect for Compressor {
    fn process(&mut self, input: f32) -> f32 {
        // Get envelope level (peak detection)
        let envelope = self.envelope_follower.process(input);

        // Convert to dB
        let envelope_db = linear_to_db(envelope);

        // Calculate gain reduction
        let gain_reduction_db = self.gain_computer.compute_gain_db(envelope_db);

        // Convert to linear and apply
        let gain_linear = db_to_linear(gain_reduction_db);

        // Apply makeup gain (smoothed)
        let makeup = self.makeup_gain.next();

        input * gain_linear * makeup
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(
            input.len(),
            output.len(),
            "Input and output buffers must have the same length"
        );

        for (in_sample, out_sample) in input.iter().zip(output.iter_mut()) {
            *out_sample = self.process(*in_sample);
        }
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;

        // Rebuild envelope follower with new sample rate
        let attack_coeff = self.envelope_follower.attack_coeff;
        let release_coeff = self.envelope_follower.release_coeff;
        self.envelope_follower = EnvelopeFollower::new(sample_rate);
        self.envelope_follower.attack_coeff = attack_coeff;
        self.envelope_follower.release_coeff = release_coeff;

        self.makeup_gain.set_sample_rate(sample_rate);
    }

    fn reset(&mut self) {
        self.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressor_basic() {
        let mut comp = Compressor::new(44100.0);
        comp.set_threshold_db(-20.0);
        comp.set_ratio(4.0);
        comp.reset();

        // Process some samples
        for i in 0..100 {
            let input = (i as f32 * 0.01).sin();
            let output = comp.process(input);
            assert!(output.is_finite());
        }
    }

    #[test]
    fn test_compressor_reduces_peaks() {
        let mut comp = Compressor::new(44100.0);
        comp.set_threshold_db(-20.0);
        comp.set_ratio(4.0);
        comp.set_attack_ms(1.0); // Fast attack
        comp.set_release_ms(50.0);
        comp.reset();

        // Generate a loud signal (above threshold)
        let loud_input = 0.5; // Roughly -6dB

        // Process enough samples for envelope to settle
        let mut output = 0.0;
        for _ in 0..1000 {
            output = comp.process(loud_input);
        }

        // Output should be quieter than input (compressed)
        assert!(output.abs() < loud_input);
    }

    #[test]
    fn test_compressor_passthrough_quiet() {
        let mut comp = Compressor::new(44100.0);
        comp.set_threshold_db(-20.0);
        comp.set_ratio(4.0);
        comp.reset();

        // Generate a quiet signal (below threshold)
        let quiet_input = 0.01; // Roughly -40dB

        // Process some samples
        let mut output = 0.0;
        for _ in 0..100 {
            output = comp.process(quiet_input);
        }

        // Quiet signals should pass through mostly unchanged
        assert!((output - quiet_input).abs() < 0.001);
    }

    #[test]
    fn test_compressor_makeup_gain() {
        let mut comp = Compressor::new(44100.0);
        comp.set_threshold_db(-20.0);
        comp.set_ratio(4.0);
        comp.set_makeup_gain_db(12.0); // +12dB boost
        comp.reset();

        let input = 0.1;

        // Process samples to let makeup gain settle
        let mut output = 0.0;
        for _ in 0..1000 {
            output = comp.process(input);
        }

        // Output should be boosted
        assert!(output.abs() > input);
    }

    #[test]
    fn test_compressor_parameter_clamping() {
        let mut comp = Compressor::new(44100.0);

        // Test extreme values don't cause issues
        comp.set_threshold_db(-100.0); // Should clamp
        comp.set_ratio(100.0); // Should clamp
        comp.set_attack_ms(0.01); // Should clamp
        comp.set_release_ms(10000.0); // Should clamp
        comp.reset();

        // Should not panic or produce invalid output
        for _ in 0..100 {
            let output = comp.process(0.5);
            assert!(output.is_finite());
        }
    }

    #[test]
    fn test_compressor_block_processing() {
        let mut comp = Compressor::new(44100.0);
        comp.set_threshold_db(-20.0);
        comp.set_ratio(4.0);

        let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut output = vec![0.0; 128];

        comp.process_block(&input, &mut output);

        // All outputs should be finite
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_db_conversion_roundtrip() {
        let linear = 0.5;
        let db = linear_to_db(linear);
        let back = db_to_linear(db);
        assert!((back - linear).abs() < 0.0001);
    }
}
