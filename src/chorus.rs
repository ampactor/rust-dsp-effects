//! Classic chorus effect with dual voices.
//!
//! Chorus creates a thick, shimmering sound by mixing the input signal with
//! modulated delayed copies. This implementation uses two delay lines with
//! phase-offset LFOs to create natural-sounding width and movement.

use crate::delay_line::InterpolatedDelay;
use crate::effect::Effect;
use crate::lfo::Lfo;
use crate::param::SmoothedParam;

/// Chorus effect with dual voices.
///
/// Creates a classic analog-style chorus by combining:
/// - Two modulated delay lines with phase-offset LFOs
/// - Linear interpolation for smooth delay time changes
/// - Smoothed parameter updates to avoid zipper noise
///
/// # Parameters
///
/// - **Rate** (0.1 - 10 Hz): LFO frequency controlling modulation speed
/// - **Depth** (0 - 1): Modulation depth as fraction of max delay
/// - **Mix** (0 - 1): Wet/dry blend (0 = dry, 1 = wet only)
///
/// # Example
///
/// ```
/// use rust_dsp_effects::{Chorus, Effect};
///
/// let mut chorus = Chorus::new(44100.0);
/// chorus.set_rate(2.0);   // 2 Hz modulation
/// chorus.set_depth(0.7);  // 70% depth
/// chorus.set_mix(0.5);    // 50/50 mix
///
/// let output = chorus.process(input);
/// ```
#[derive(Debug, Clone)]
pub struct Chorus {
    /// First voice delay line
    delay1: InterpolatedDelay,
    /// Second voice delay line
    delay2: InterpolatedDelay,
    /// LFO for first voice
    lfo1: Lfo,
    /// LFO for second voice
    lfo2: Lfo,

    /// Base delay time in samples (center point of modulation)
    base_delay_samples: f32,
    /// Maximum modulation range in samples (depth range)
    max_mod_samples: f32,

    /// LFO rate parameter (Hz)
    rate: SmoothedParam,
    /// Modulation depth parameter (0-1)
    depth: SmoothedParam,
    /// Wet/dry mix parameter (0-1)
    mix: SmoothedParam,

    /// Sample rate
    sample_rate: f32,
}

impl Chorus {
    /// Creates a new chorus effect.
    ///
    /// # Default Parameters
    ///
    /// - Base delay: 15ms
    /// - Max modulation: ±5ms
    /// - Rate: 1.0 Hz
    /// - Depth: 0.5
    /// - Mix: 0.5
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Audio sample rate in Hz
    pub fn new(sample_rate: f32) -> Self {
        const BASE_DELAY_MS: f32 = 15.0;
        const MAX_MOD_MS: f32 = 5.0;
        const MAX_DELAY_MS: f32 = BASE_DELAY_MS + MAX_MOD_MS;

        let base_delay_samples = (BASE_DELAY_MS / 1000.0) * sample_rate;
        let max_mod_samples = (MAX_MOD_MS / 1000.0) * sample_rate;
        let max_delay_samples = ((MAX_DELAY_MS / 1000.0) * sample_rate).ceil() as usize;

        let mut lfo1 = Lfo::new(sample_rate);
        let mut lfo2 = Lfo::new(sample_rate);

        // Phase offset the second LFO by 90° for stereo-like width
        lfo2.set_phase(0.25);

        // Default rate of 1 Hz
        lfo1.set_frequency(1.0);
        lfo2.set_frequency(1.0);

        Self {
            delay1: InterpolatedDelay::new(max_delay_samples),
            delay2: InterpolatedDelay::new(max_delay_samples),
            lfo1,
            lfo2,
            base_delay_samples,
            max_mod_samples,
            rate: SmoothedParam::with_config(1.0, sample_rate, 10.0),
            depth: SmoothedParam::with_config(0.5, sample_rate, 10.0),
            mix: SmoothedParam::with_config(0.5, sample_rate, 10.0),
            sample_rate,
        }
    }

    /// Sets the LFO rate in Hz.
    ///
    /// Typical range: 0.1 to 10 Hz
    /// Lower values = slow, subtle movement
    /// Higher values = fast, vibrato-like effect
    pub fn set_rate(&mut self, rate_hz: f32) {
        self.rate.set_target(rate_hz.clamp(0.1, 10.0));
    }

    /// Sets the modulation depth (0-1).
    ///
    /// 0.0 = no modulation (just delay)
    /// 1.0 = full ±5ms modulation range
    pub fn set_depth(&mut self, depth: f32) {
        self.depth.set_target(depth.clamp(0.0, 1.0));
    }

    /// Sets the wet/dry mix (0-1).
    ///
    /// 0.0 = dry signal only (bypass)
    /// 1.0 = wet signal only (no dry)
    /// 0.5 = equal mix (typical)
    pub fn set_mix(&mut self, mix: f32) {
        self.mix.set_target(mix.clamp(0.0, 1.0));
    }

    /// Resets the effect state.
    ///
    /// Clears delay buffers and resets LFOs to initial phase.
    pub fn reset(&mut self) {
        self.delay1.clear();
        self.delay2.clear();
        self.lfo1.reset();
        self.lfo2.reset();
    }
}

impl Effect for Chorus {
    fn process(&mut self, input: f32) -> f32 {
        // Update smoothed parameters
        let rate = self.rate.next();
        let depth = self.depth.next();
        let mix = self.mix.next();

        // Update LFO frequencies if rate changed
        self.lfo1.set_frequency(rate);
        self.lfo2.set_frequency(rate);

        // Generate LFO modulation signals [-1, 1]
        let mod1 = self.lfo1.next();
        let mod2 = self.lfo2.next();

        // Calculate delay times
        // delay = base_delay + (lfo * depth * max_mod)
        let delay_time1 = self.base_delay_samples + (mod1 * depth * self.max_mod_samples);
        let delay_time2 = self.base_delay_samples + (mod2 * depth * self.max_mod_samples);

        // Read delayed samples
        let wet1 = self.delay1.read(delay_time1);
        let wet2 = self.delay2.read(delay_time2);

        // Write input to both delay lines
        self.delay1.write(input);
        self.delay2.write(input);

        // Mix: average both voices and blend with dry signal
        let wet = (wet1 + wet2) * 0.5;
        input * (1.0 - mix) + wet * mix
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

        // Recalculate delay times in samples
        const BASE_DELAY_MS: f32 = 15.0;
        const MAX_MOD_MS: f32 = 5.0;

        self.base_delay_samples = (BASE_DELAY_MS / 1000.0) * sample_rate;
        self.max_mod_samples = (MAX_MOD_MS / 1000.0) * sample_rate;

        // Update LFOs and parameters
        self.lfo1.set_sample_rate(sample_rate);
        self.lfo2.set_sample_rate(sample_rate);
        self.rate.set_sample_rate(sample_rate);
        self.depth.set_sample_rate(sample_rate);
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
    fn test_chorus_basic() {
        let mut chorus = Chorus::new(44100.0);
        chorus.set_mix(1.0); // Full wet for testing

        // Process some samples
        for i in 0..1000 {
            let input = (i as f32 * 0.001).sin();
            let output = chorus.process(input);

            // Output should be in valid range
            assert!(output.is_finite());
            assert!(output.abs() <= 2.0); // Allow some headroom
        }
    }

    #[test]
    fn test_chorus_bypass() {
        let mut chorus = Chorus::new(44100.0);
        chorus.set_mix(0.0); // Full dry (bypass)

        // Process enough samples to let parameter smoothing fully settle
        // Smoothing time is 10ms = ~440 samples at 44.1kHz
        // Need to process with non-zero input to fill delay buffers properly
        for _ in 0..1000 {
            chorus.process(1.0);
        }

        let input = 0.5;
        let output = chorus.process(input);

        // With mix ~0.0, output should be very close to input
        // Small tolerance for floating point smoothing
        assert!((output - input).abs() < 0.05);
    }

    #[test]
    fn test_chorus_parameter_ranges() {
        let mut chorus = Chorus::new(44100.0);

        // Test clamping
        chorus.set_rate(100.0); // Should clamp to 10 Hz
        chorus.set_depth(2.0); // Should clamp to 1.0
        chorus.set_mix(-1.0); // Should clamp to 0.0

        // Should not panic or produce invalid output
        let output = chorus.process(1.0);
        assert!(output.is_finite());
    }

    #[test]
    fn test_chorus_reset() {
        let mut chorus = Chorus::new(44100.0);

        // Process some samples
        for _ in 0..100 {
            chorus.process(1.0);
        }

        // Reset
        chorus.reset();

        // Next output should be close to input (empty delays)
        chorus.set_mix(0.5);
        let output = chorus.process(1.0);
        assert!(output.abs() < 1.0);
    }

    #[test]
    fn test_chorus_block_processing() {
        let mut chorus = Chorus::new(44100.0);

        let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut output = vec![0.0; 128];

        chorus.process_block(&input, &mut output);

        // All outputs should be finite
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_chorus_sample_rate_change() {
        let mut chorus = Chorus::new(44100.0);

        // Process at 44.1kHz
        let output1 = chorus.process(1.0);

        // Change sample rate
        chorus.set_sample_rate(48000.0);

        // Should still produce valid output
        let output2 = chorus.process(1.0);
        assert!(output1.is_finite());
        assert!(output2.is_finite());
    }
}
