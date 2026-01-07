//! Classic delay effect with feedback control.
//!
//! Provides tape-style delay with adjustable delay time, feedback, and mix.
//! Features a feedback loop for regenerating echoes and smooth parameter
//! changes to avoid zipper noise.

use crate::delay_line::InterpolatedDelay;
use crate::effect::Effect;
use crate::param::SmoothedParam;

/// Classic delay effect with feedback.
///
/// Implements a tape-style delay line with:
/// - Adjustable delay time (up to 2 seconds by default)
/// - Feedback control for regenerating echoes
/// - Wet/dry mix control
/// - Smoothed parameters for artifact-free changes
///
/// # Topology
///
/// ```text
/// input → (+) → delay_line → mix → output
///          ↑                   ↓
///          └──── feedback ←────┘
/// ```
///
/// # Example
///
/// ```
/// use rust_dsp_effects::{Delay, Effect};
///
/// let mut delay = Delay::new(44100.0);
/// delay.set_delay_time_ms(375.0);  // 1/8 note at 120 BPM
/// delay.set_feedback(0.5);          // 50% feedback
/// delay.set_mix(0.3);               // 30% wet
///
/// let output = delay.process(input);
/// ```
#[derive(Debug, Clone)]
pub struct Delay {
    /// Delay line
    delay_line: InterpolatedDelay,

    /// Maximum delay time in samples
    max_delay_samples: f32,

    /// Delay time parameter (in samples)
    delay_time: SmoothedParam,
    /// Feedback amount parameter (0-1)
    feedback: SmoothedParam,
    /// Wet/dry mix parameter (0-1)
    mix: SmoothedParam,

    /// Sample rate
    sample_rate: f32,
}

impl Delay {
    /// Creates a new delay effect with default 2-second maximum delay.
    ///
    /// # Default Parameters
    ///
    /// - Delay time: 500ms
    /// - Feedback: 0.3
    /// - Mix: 0.5
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Audio sample rate in Hz
    pub fn new(sample_rate: f32) -> Self {
        Self::with_max_delay_ms(sample_rate, 2000.0)
    }

    /// Creates a new delay effect with a custom maximum delay time.
    ///
    /// Useful for creating specific delay types:
    /// - Short delays (< 50ms): Doubling, slapback
    /// - Medium delays (50-500ms): Standard echo
    /// - Long delays (> 500ms): Ambient, soundscape
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `max_delay_ms` - Maximum delay time in milliseconds
    pub fn with_max_delay_ms(sample_rate: f32, max_delay_ms: f32) -> Self {
        let max_delay_samples = ((max_delay_ms / 1000.0) * sample_rate).ceil() as usize;
        let max_delay_samples_f32 = max_delay_samples as f32;

        // Default to 500ms delay time
        let default_delay_samples = ((500.0 / 1000.0) * sample_rate).min(max_delay_samples_f32);

        Self {
            delay_line: InterpolatedDelay::new(max_delay_samples),
            max_delay_samples: max_delay_samples_f32,
            delay_time: SmoothedParam::with_config(default_delay_samples, sample_rate, 50.0),
            feedback: SmoothedParam::with_config(0.3, sample_rate, 10.0),
            mix: SmoothedParam::with_config(0.5, sample_rate, 10.0),
            sample_rate,
        }
    }

    /// Sets the delay time in milliseconds.
    ///
    /// The delay time will be clamped to the maximum delay set during construction.
    ///
    /// Typical values:
    /// - Slapback: 75-120ms
    /// - Eighth note at 120 BPM: 250ms
    /// - Quarter note at 120 BPM: 500ms
    pub fn set_delay_time_ms(&mut self, delay_ms: f32) {
        let delay_samples = (delay_ms / 1000.0) * self.sample_rate;
        let clamped = delay_samples.clamp(1.0, self.max_delay_samples - 1.0);
        self.delay_time.set_target(clamped);
    }

    /// Sets the feedback amount (0-1).
    ///
    /// Controls how much of the delayed signal is fed back into the delay line.
    /// - 0.0 = single echo (no feedback)
    /// - 0.5 = moderate repeats
    /// - 0.95 = maximum safe feedback (many repeats)
    ///
    /// Values above 0.95 are clamped to prevent runaway feedback.
    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback.set_target(feedback.clamp(0.0, 0.95));
    }

    /// Sets the wet/dry mix (0-1).
    ///
    /// - 0.0 = dry signal only (bypass)
    /// - 0.5 = equal mix (typical)
    /// - 1.0 = wet signal only (no dry)
    pub fn set_mix(&mut self, mix: f32) {
        self.mix.set_target(mix.clamp(0.0, 1.0));
    }

    /// Resets the effect state.
    ///
    /// Clears the delay buffer and snaps parameters to their targets.
    pub fn reset(&mut self) {
        self.delay_line.clear();
        // Snap parameters to targets (no smoothing)
        let delay_target = self.delay_time.target();
        let feedback_target = self.feedback.target();
        let mix_target = self.mix.target();
        self.delay_time.set_immediate(delay_target);
        self.feedback.set_immediate(feedback_target);
        self.mix.set_immediate(mix_target);
    }
}

impl Effect for Delay {
    fn process(&mut self, input: f32) -> f32 {
        // Update smoothed parameters
        let delay_samples = self.delay_time.next();
        let feedback = self.feedback.next();
        let mix = self.mix.next();

        // Read delayed signal
        let delayed = self.delay_line.read(delay_samples);

        // Apply feedback: input + (delayed * feedback)
        let feedback_signal = input + (delayed * feedback);

        // Write to delay line
        self.delay_line.write(feedback_signal);

        // Mix dry and wet signals
        input * (1.0 - mix) + delayed * mix
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

        // Recalculate max delay in samples
        let max_delay_ms = (self.max_delay_samples / self.sample_rate) * 1000.0;
        self.max_delay_samples = ((max_delay_ms / 1000.0) * sample_rate).ceil();

        // Update parameter smoothing
        self.delay_time.set_sample_rate(sample_rate);
        self.feedback.set_sample_rate(sample_rate);
        self.mix.set_sample_rate(sample_rate);
    }

    fn reset(&mut self) {
        self.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_basic() {
        let mut delay = Delay::new(44100.0);
        delay.set_delay_time_ms(100.0); // 100ms delay
        delay.set_feedback(0.0); // No feedback for simple test
        delay.set_mix(1.0); // Full wet
        delay.reset();

        // Process impulse
        let output = delay.process(1.0);
        assert_eq!(output, 0.0); // First output is silence (delay not filled yet)

        // Process silence and look for the delayed impulse
        // Should appear around 4410 samples (100ms at 44.1kHz)
        let mut max_output: f32 = 0.0;
        for _ in 0..4500 {
            let out = delay.process(0.0);
            max_output = max_output.max(out);
        }

        // Should have seen the impulse at some point
        assert!(max_output > 0.9);
    }

    #[test]
    fn test_delay_feedback() {
        let mut delay = Delay::new(44100.0);
        delay.set_delay_time_ms(10.0); // Short delay for testing
        delay.set_feedback(0.5); // 50% feedback
        delay.set_mix(1.0); // Full wet
        delay.reset();

        // Process impulse
        delay.process(1.0);

        // Process silence for delay time
        let delay_samples = ((10.0 / 1000.0) * 44100.0) as usize;
        for _ in 0..delay_samples {
            delay.process(0.0);
        }

        // Should see delayed signal
        let first_echo = delay.process(0.0);
        assert!(first_echo > 0.9); // Close to 1.0

        // Process more silence for another delay period
        for _ in 0..delay_samples {
            delay.process(0.0);
        }

        // Should see feedback echo (roughly 0.5 of original)
        let second_echo = delay.process(0.0);
        assert!(second_echo > 0.4 && second_echo < 0.6);
    }

    #[test]
    fn test_delay_bypass() {
        let mut delay = Delay::new(44100.0);
        delay.set_mix(0.0); // Full dry (bypass)
        delay.reset();

        let input = 0.5;
        let output = delay.process(input);

        // Should pass through unchanged
        assert!((output - input).abs() < 0.001);
    }

    #[test]
    fn test_delay_parameter_clamping() {
        let mut delay = Delay::new(44100.0);

        // Test feedback clamping
        delay.set_feedback(1.5); // Should clamp to 0.95
        delay.reset();

        // Process some samples - should not explode
        for _ in 0..1000 {
            let output = delay.process(0.1);
            assert!(output.is_finite());
            assert!(output.abs() < 10.0); // Reasonable bounds
        }
    }

    #[test]
    fn test_delay_max_time() {
        let mut delay = Delay::with_max_delay_ms(44100.0, 100.0);

        // Try to set delay beyond max
        delay.set_delay_time_ms(500.0); // Should clamp to ~100ms
        delay.reset();

        // Should not panic
        for _ in 0..5000 {
            delay.process(0.1);
        }
    }

    #[test]
    fn test_delay_clear() {
        let mut delay = Delay::new(44100.0);
        delay.set_delay_time_ms(10.0);
        delay.set_mix(1.0);

        // Fill delay with signal
        for _ in 0..1000 {
            delay.process(1.0);
        }

        // Reset
        delay.reset();

        // Next output should be close to zero (empty delay)
        let output = delay.process(0.0);
        assert!((output - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_delay_block_processing() {
        let mut delay = Delay::new(44100.0);
        delay.set_delay_time_ms(50.0);
        delay.set_feedback(0.3);
        delay.set_mix(0.5);

        let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut output = vec![0.0; 128];

        delay.process_block(&input, &mut output);

        // All outputs should be finite
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_delay_sample_rate_change() {
        let mut delay = Delay::new(44100.0);
        delay.set_delay_time_ms(100.0);

        // Process at 44.1kHz
        let output1 = delay.process(1.0);

        // Change sample rate
        delay.set_sample_rate(48000.0);

        // Should still produce valid output
        let output2 = delay.process(1.0);
        assert!(output1.is_finite());
        assert!(output2.is_finite());
    }
}
