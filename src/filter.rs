//! Low-pass filter effect using biquad topology.
//!
//! Provides a smooth low-pass filter for removing high-frequency content,
//! useful for tone shaping, anti-aliasing, and subtractive synthesis.

use crate::biquad::{lowpass_coefficients, Biquad};
use crate::effect::Effect;
use crate::param::SmoothedParam;

/// Low-pass filter effect with smoothed parameter control.
///
/// Implements a second-order (12 dB/octave) low-pass filter using
/// biquad topology with RBJ cookbook coefficients.
///
/// # Parameters
///
/// - **Cutoff Frequency**: Controls where high frequencies begin to roll off
/// - **Q Factor (Resonance)**: Controls the sharpness of the cutoff
///   - 0.707: Butterworth (maximally flat)
///   - > 1.0: Resonant peak at cutoff
///   - < 0.707: Gentler rolloff
///
/// # Example
///
/// ```
/// use rust_dsp_effects::{LowPassFilter, Effect};
///
/// let mut filter = LowPassFilter::new(44100.0);
/// filter.set_cutoff_hz(1000.0);  // 1kHz cutoff
/// filter.set_q(0.707);            // Butterworth response
///
/// let output = filter.process(input);
/// ```
#[derive(Debug, Clone)]
pub struct LowPassFilter {
    /// Biquad filter core
    biquad: Biquad,

    /// Cutoff frequency parameter (Hz)
    cutoff: SmoothedParam,
    /// Q factor parameter
    q: SmoothedParam,

    /// Sample rate
    sample_rate: f32,

    /// Flag to trigger coefficient recalculation
    needs_update: bool,
}

impl LowPassFilter {
    /// Creates a new low-pass filter.
    ///
    /// # Default Parameters
    ///
    /// - Cutoff: 1000 Hz
    /// - Q: 0.707 (Butterworth)
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Audio sample rate in Hz
    pub fn new(sample_rate: f32) -> Self {
        let mut filter = Self {
            biquad: Biquad::new(),
            cutoff: SmoothedParam::with_config(1000.0, sample_rate, 20.0),
            q: SmoothedParam::with_config(0.707, sample_rate, 20.0),
            sample_rate,
            needs_update: true,
        };

        // Initialize coefficients
        filter.update_coefficients();
        filter
    }

    /// Sets the cutoff frequency in Hz.
    ///
    /// The cutoff is the -3dB point where high frequencies start to roll off.
    ///
    /// Typical ranges:
    /// - 20-200 Hz: Sub bass filtering
    /// - 200-2000 Hz: Bass/low-mid emphasis
    /// - 2000-20000 Hz: High-frequency rolloff
    pub fn set_cutoff_hz(&mut self, cutoff: f32) {
        let clamped = cutoff.clamp(20.0, self.sample_rate * 0.49);
        self.cutoff.set_target(clamped);
        self.needs_update = true;
    }

    /// Sets the Q factor (resonance).
    ///
    /// Controls the sharpness of the filter response at the cutoff frequency.
    ///
    /// Common values:
    /// - 0.5: Very gentle rolloff
    /// - 0.707: Butterworth (maximally flat passband)
    /// - 1.0: Slight resonant peak
    /// - 2.0+: Pronounced resonance (typical for synthesizer filters)
    pub fn set_q(&mut self, q: f32) {
        let clamped = q.clamp(0.1, 20.0);
        self.q.set_target(clamped);
        self.needs_update = true;
    }

    /// Resets the filter state.
    ///
    /// Clears the biquad delay lines and snaps parameters to targets.
    pub fn reset(&mut self) {
        self.biquad.clear();
        let cutoff_target = self.cutoff.target();
        let q_target = self.q.target();
        self.cutoff.set_immediate(cutoff_target);
        self.q.set_immediate(q_target);
        self.needs_update = true;
        self.update_coefficients();
    }

    /// Updates the biquad coefficients based on current parameters.
    fn update_coefficients(&mut self) {
        let cutoff = self.cutoff.get();
        let q = self.q.get();

        let (b0, b1, b2, a0, a1, a2) = lowpass_coefficients(cutoff, q, self.sample_rate);
        self.biquad.set_coefficients(b0, b1, b2, a0, a1, a2);
        self.needs_update = false;
    }
}

impl Effect for LowPassFilter {
    fn process(&mut self, input: f32) -> f32 {
        // Update smoothed parameters
        let _cutoff = self.cutoff.next();
        let _q = self.q.next();

        // Check if coefficients need updating
        // We update when parameters are changing to balance smoothness vs CPU cost
        if self.needs_update || !self.cutoff.is_settled() || !self.q.is_settled() {
            self.update_coefficients();
        }

        self.biquad.process(input)
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
        self.cutoff.set_sample_rate(sample_rate);
        self.q.set_sample_rate(sample_rate);
        self.needs_update = true;
        self.update_coefficients();
    }

    fn reset(&mut self) {
        self.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lowpass_basic() {
        let mut filter = LowPassFilter::new(44100.0);
        filter.set_cutoff_hz(1000.0);
        filter.set_q(0.707);
        filter.reset();

        // Process some samples
        for i in 0..100 {
            let input = (i as f32 * 0.1).sin();
            let output = filter.process(input);
            assert!(output.is_finite());
        }
    }

    #[test]
    fn test_lowpass_dc_response() {
        let mut filter = LowPassFilter::new(44100.0);
        filter.set_cutoff_hz(1000.0);
        filter.set_q(0.707);
        filter.reset();

        // DC (0 Hz) should pass through a low-pass filter with unity gain
        let mut output = 0.0;
        for _ in 0..1000 {
            output = filter.process(1.0);
        }

        // Should converge to approximately 1.0
        assert!((output - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_lowpass_high_freq_attenuation() {
        let mut filter = LowPassFilter::new(44100.0);
        filter.set_cutoff_hz(100.0); // Very low cutoff
        filter.set_q(0.707);
        filter.reset();

        // Generate high-frequency signal (10kHz)
        let freq = 10000.0;
        let mut sum = 0.0;
        for i in 0..1000 {
            let t = i as f32 / 44100.0;
            let input = (2.0 * std::f32::consts::PI * freq * t).sin();
            let output = filter.process(input);
            sum += output.abs();
        }

        let avg = sum / 1000.0;

        // High frequencies should be heavily attenuated
        assert!(avg < 0.1);
    }

    #[test]
    fn test_lowpass_parameter_clamping() {
        let mut filter = LowPassFilter::new(44100.0);

        // Test extreme values don't cause issues
        filter.set_cutoff_hz(100000.0); // Should clamp to Nyquist
        filter.set_q(100.0); // Should clamp to 20.0
        filter.reset();

        // Should not panic or produce invalid output
        for _ in 0..100 {
            let output = filter.process(0.1);
            assert!(output.is_finite());
        }
    }

    #[test]
    fn test_lowpass_reset() {
        let mut filter = LowPassFilter::new(44100.0);

        // Process some samples to fill state
        for _ in 0..100 {
            filter.process(1.0);
        }

        // Reset
        filter.reset();

        // Next output should respond to new input cleanly
        let output = filter.process(0.0);
        assert!(output.abs() < 0.1); // Should be close to zero
    }

    #[test]
    fn test_lowpass_block_processing() {
        let mut filter = LowPassFilter::new(44100.0);
        filter.set_cutoff_hz(2000.0);
        filter.set_q(1.0);

        let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut output = vec![0.0; 128];

        filter.process_block(&input, &mut output);

        // All outputs should be finite
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_lowpass_sample_rate_change() {
        let mut filter = LowPassFilter::new(44100.0);
        filter.set_cutoff_hz(1000.0);

        // Process at 44.1kHz
        let output1 = filter.process(1.0);

        // Change sample rate
        filter.set_sample_rate(48000.0);

        // Should still produce valid output
        let output2 = filter.process(1.0);
        assert!(output1.is_finite());
        assert!(output2.is_finite());
    }
}
