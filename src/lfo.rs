//! Low Frequency Oscillator (LFO) for modulation effects.
//!
//! Provides smooth, periodic modulation signals used in chorus, flanger,
//! tremolo, and other time-based effects.

#[cfg(not(feature = "std"))]
use core::f32::consts::PI;
#[cfg(feature = "std")]
use std::f32::consts::PI;

/// Low Frequency Oscillator for generating modulation signals.
///
/// Generates a sine wave at sub-audio frequencies (typically 0.1-20 Hz).
/// Uses phase accumulation for efficient, alias-free oscillation.
///
/// # Example
///
/// ```
/// use rust_dsp_effects::Lfo;
///
/// let mut lfo = Lfo::new(44100.0);
/// lfo.set_frequency(2.0); // 2 Hz modulation
///
/// // Generate modulation values in [-1.0, 1.0]
/// let value = lfo.next();
/// ```
#[derive(Debug, Clone)]
pub struct Lfo {
    /// Current phase position [0.0, 1.0)
    phase: f32,
    /// Phase increment per sample
    phase_increment: f32,
    /// Sample rate in Hz
    sample_rate: f32,
}

impl Lfo {
    /// Creates a new LFO with the given sample rate.
    ///
    /// Initial frequency is 0 Hz (no modulation).
    pub fn new(sample_rate: f32) -> Self {
        Self {
            phase: 0.0,
            phase_increment: 0.0,
            sample_rate,
        }
    }

    /// Sets the LFO frequency in Hz.
    ///
    /// Typical range: 0.1 to 20 Hz
    /// For chorus: 0.5 to 5 Hz
    pub fn set_frequency(&mut self, freq_hz: f32) {
        self.phase_increment = freq_hz / self.sample_rate;
    }

    /// Sets the initial phase offset [0.0, 1.0].
    ///
    /// Useful for creating phase-offset LFOs in multi-voice effects.
    /// 0.0 = 0°, 0.25 = 90°, 0.5 = 180°, 0.75 = 270°
    pub fn set_phase(&mut self, phase: f32) {
        self.phase = phase.clamp(0.0, 1.0);
    }

    /// Generates the next modulation sample.
    ///
    /// Returns a value in the range [-1.0, 1.0].
    #[inline]
    pub fn next(&mut self) -> f32 {
        let output = (self.phase * 2.0 * PI).sin();

        // Advance phase and wrap
        self.phase += self.phase_increment;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        output
    }

    /// Resets the LFO to initial phase.
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    /// Updates the sample rate.
    ///
    /// Call this when the audio system sample rate changes.
    /// Preserves the current frequency setting.
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        let freq = self.phase_increment * self.sample_rate;
        self.sample_rate = sample_rate;
        self.phase_increment = freq / sample_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lfo_phase_accumulation() {
        let mut lfo = Lfo::new(44100.0);
        lfo.set_frequency(1.0); // 1 Hz = one cycle per second

        // After 44100 samples (1 second), should complete one cycle
        for _ in 0..44100 {
            lfo.next();
        }

        // Phase should be very close to 0 or 1 (wrapped around)
        // Due to floating point precision, allow some tolerance
        let phase_error = (lfo.phase - 0.0).abs().min((lfo.phase - 1.0).abs());
        assert!(phase_error < 0.01);
    }

    #[test]
    fn test_lfo_output_range() {
        let mut lfo = Lfo::new(44100.0);
        lfo.set_frequency(5.0);

        // Check that output stays in [-1.0, 1.0]
        for _ in 0..1000 {
            let value = lfo.next();
            assert!(value >= -1.0 && value <= 1.0);
        }
    }

    #[test]
    fn test_lfo_phase_offset() {
        let mut lfo1 = Lfo::new(44100.0);
        let mut lfo2 = Lfo::new(44100.0);

        lfo1.set_frequency(2.0);
        lfo2.set_frequency(2.0);
        lfo2.set_phase(0.5); // 180° offset

        let val1 = lfo1.next();
        let val2 = lfo2.next();

        // Should be approximately opposite
        assert!((val1 + val2).abs() < 0.01);
    }

    #[test]
    fn test_lfo_sample_rate_change() {
        let mut lfo = Lfo::new(44100.0);
        lfo.set_frequency(440.0);

        let phase_inc_44k = lfo.phase_increment;

        lfo.set_sample_rate(48000.0);
        let phase_inc_48k = lfo.phase_increment;

        // Phase increment should scale inversely with sample rate
        // Higher sample rate = smaller phase increment for same frequency
        let ratio = 48000.0 / 44100.0;
        assert!((phase_inc_44k / phase_inc_48k - ratio).abs() < 0.0001);
    }
}
