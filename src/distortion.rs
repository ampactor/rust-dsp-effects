//! Distortion effects with multiple waveshaping algorithms.
//!
//! This module provides classic distortion/overdrive effects suitable for
//! guitar and synthesizer processing. The [`Distortion`] effect combines:
//!
//! - **Drive**: Input gain before waveshaping (controls distortion amount)
//! - **Waveshaping**: Non-linear transfer function (tanh, hard clip, etc.)
//! - **Tone**: Post-distortion lowpass filter (tames harsh harmonics)
//! - **Level**: Output gain for volume matching
//!
//! ## Waveshaping Algorithms
//!
//! Different waveshaping functions produce different harmonic content:
//!
//! - [`WaveShape::SoftClip`]: Hyperbolic tangent - smooth, tube-like saturation
//! - [`WaveShape::HardClip`]: Brick-wall limiting - aggressive, transistor-like
//! - [`WaveShape::Foldback`]: Folds signal back - synth-style, rich harmonics
//! - [`WaveShape::Asymmetric`]: Tube-like asymmetric clipping - even harmonics
//!
//! ## Example
//!
//! ```rust
//! use rust_dsp_effects::{Effect, Distortion, WaveShape};
//!
//! let mut dist = Distortion::new(48000.0);
//! dist.set_drive_db(20.0);      // 20dB of gain into waveshaper
//! dist.set_tone_hz(4000.0);     // 4kHz lowpass
//! dist.set_level_db(-12.0);     // -12dB output to compensate
//! dist.set_waveshape(WaveShape::SoftClip);
//!
//! // Process audio
//! let input = 0.1;  // Clean guitar signal
//! let output = dist.process(input);
//! ```
//!
//! ## Anti-Aliasing Note
//!
//! Waveshaping creates harmonics that can alias when they exceed Nyquist.
//! For highest quality, wrap the distortion in an oversampling processor
//! (to be implemented). For many use cases, the built-in tone control
//! provides sufficient alias suppression.

use crate::param::SmoothedParam;
use crate::Effect;

#[cfg(not(feature = "std"))]
use libm::{expf, fabsf, tanhf};

#[cfg(feature = "std")]
fn tanhf(x: f32) -> f32 {
    x.tanh()
}
#[cfg(feature = "std")]
fn expf(x: f32) -> f32 {
    x.exp()
}
#[cfg(feature = "std")]
fn fabsf(x: f32) -> f32 {
    x.abs()
}

/// Waveshaping algorithm selection.
///
/// Each algorithm produces different harmonic characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaveShape {
    /// Hyperbolic tangent soft clipping.
    ///
    /// Smooth saturation that approaches ±1 asymptotically.
    /// Produces primarily odd harmonics, similar to tube amplifiers.
    /// This is the most "musical" and commonly used algorithm.
    #[default]
    SoftClip,

    /// Hard clipping at ±1.
    ///
    /// Abrupt limiting that creates a flat top on the waveform.
    /// Produces harsh odd harmonics, similar to transistor distortion.
    /// Sounds more aggressive than soft clipping.
    HardClip,

    /// Foldback distortion.
    ///
    /// When signal exceeds threshold, it "folds" back instead of clipping.
    /// Creates rich, complex harmonic content. Popular in synthesizers.
    /// The `threshold` parameter controls where folding begins.
    Foldback,

    /// Asymmetric soft clipping.
    ///
    /// Clips positive and negative portions differently, similar to
    /// tube amplifiers with asymmetric transfer characteristics.
    /// Produces both even and odd harmonics for a warmer sound.
    Asymmetric,
}

/// Distortion effect with waveshaping and tone control.
///
/// A complete distortion unit with:
/// - Smoothed drive (input gain)
/// - Selectable waveshaping algorithm
/// - One-pole lowpass tone control
/// - Smoothed output level
///
/// All parameters use smoothing to prevent zipper noise during changes.
pub struct Distortion {
    // Parameters with smoothing
    drive: SmoothedParam,      // Linear gain (not dB)
    level: SmoothedParam,      // Linear gain (not dB)
    tone_coeff: SmoothedParam, // Filter coefficient

    // Current settings (for recalculation)
    sample_rate: f32,
    tone_freq_hz: f32,

    // Waveshaping
    waveshape: WaveShape,
    foldback_threshold: f32,

    // Filter state
    tone_filter_state: f32,
}

impl Distortion {
    /// Create a new distortion effect.
    ///
    /// Initializes with default settings:
    /// - Drive: 0 dB (unity gain)
    /// - Level: 0 dB (unity gain)
    /// - Tone: 8000 Hz lowpass
    /// - Waveshape: SoftClip
    pub fn new(sample_rate: f32) -> Self {
        let mut dist = Self {
            drive: SmoothedParam::with_config(1.0, sample_rate, 5.0),
            level: SmoothedParam::with_config(1.0, sample_rate, 5.0),
            tone_coeff: SmoothedParam::with_config(0.0, sample_rate, 5.0),
            sample_rate,
            tone_freq_hz: 8000.0,
            waveshape: WaveShape::default(),
            foldback_threshold: 0.8,
            tone_filter_state: 0.0,
        };
        dist.recalculate_tone_coeff();
        dist
    }

    /// Set drive amount in decibels.
    ///
    /// Drive controls the input gain before waveshaping:
    /// - 0 dB: Unity gain, minimal distortion
    /// - 10-20 dB: Moderate overdrive
    /// - 30+ dB: Heavy distortion
    ///
    /// Typical range: 0 to 40 dB
    pub fn set_drive_db(&mut self, db: f32) {
        let gain = db_to_linear(db);
        self.drive.set_target(gain);
    }

    /// Set output level in decibels.
    ///
    /// Use negative values to compensate for the volume increase
    /// from distortion. Typical range: -24 to +6 dB
    pub fn set_level_db(&mut self, db: f32) {
        let gain = db_to_linear(db);
        self.level.set_target(gain);
    }

    /// Set tone control frequency in Hz.
    ///
    /// Controls the cutoff of the post-distortion lowpass filter:
    /// - Lower values: Darker, less harsh
    /// - Higher values: Brighter, more present
    ///
    /// Typical range: 1000 to 8000 Hz
    pub fn set_tone_hz(&mut self, freq_hz: f32) {
        self.tone_freq_hz = freq_hz;
        self.recalculate_tone_coeff();
    }

    /// Set the waveshaping algorithm.
    pub fn set_waveshape(&mut self, waveshape: WaveShape) {
        self.waveshape = waveshape;
    }

    /// Set foldback threshold (only affects Foldback waveshape).
    ///
    /// Controls where the signal starts folding back.
    /// Range: 0.1 to 1.0 (default: 0.8)
    pub fn set_foldback_threshold(&mut self, threshold: f32) {
        self.foldback_threshold = threshold.clamp(0.1, 1.0);
    }

    /// Get current drive in dB.
    pub fn drive_db(&self) -> f32 {
        linear_to_db(self.drive.target())
    }

    /// Get current level in dB.
    pub fn level_db(&self) -> f32 {
        linear_to_db(self.level.target())
    }

    /// Get current tone frequency in Hz.
    pub fn tone_hz(&self) -> f32 {
        self.tone_freq_hz
    }

    /// Get current waveshape.
    pub fn get_waveshape(&self) -> WaveShape {
        self.waveshape
    }

    /// Recalculate tone filter coefficient from frequency.
    fn recalculate_tone_coeff(&mut self) {
        // One-pole lowpass coefficient
        // coeff = 1 - exp(-2π * fc / fs)
        let normalized = self.tone_freq_hz / self.sample_rate;
        let coeff = 1.0 - expf(-core::f32::consts::TAU * normalized);
        self.tone_coeff.set_target(coeff);
    }

    /// Apply waveshaping to a sample.
    #[inline]
    fn apply_waveshape(&self, x: f32) -> f32 {
        match self.waveshape {
            WaveShape::SoftClip => soft_clip(x),
            WaveShape::HardClip => hard_clip(x),
            WaveShape::Foldback => foldback(x, self.foldback_threshold),
            WaveShape::Asymmetric => asymmetric_clip(x),
        }
    }

    /// One-pole lowpass filter for tone control.
    #[inline]
    fn tone_filter(&mut self, input: f32, coeff: f32) -> f32 {
        self.tone_filter_state += coeff * (input - self.tone_filter_state);
        self.tone_filter_state
    }
}

impl Effect for Distortion {
    fn process(&mut self, input: f32) -> f32 {
        // Get smoothed parameters
        let drive = self.drive.next();
        let level = self.level.next();
        let tone_coeff = self.tone_coeff.next();

        // Apply drive
        let driven = input * drive;

        // Waveshaping
        let shaped = self.apply_waveshape(driven);

        // Tone filter (lowpass)
        let filtered = self.tone_filter(shaped, tone_coeff);

        // Output level
        filtered * level
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.drive.set_sample_rate(sample_rate);
        self.level.set_sample_rate(sample_rate);
        self.tone_coeff.set_sample_rate(sample_rate);
        self.recalculate_tone_coeff();
    }

    fn reset(&mut self) {
        self.tone_filter_state = 0.0;
        self.drive.snap_to_target();
        self.level.snap_to_target();
        self.tone_coeff.snap_to_target();
    }
}

// ============================================================================
// Waveshaping Functions
// ============================================================================

/// Soft clip using hyperbolic tangent.
///
/// Smooth saturation: tanh(x) approaches ±1 asymptotically.
/// Produces primarily odd harmonics.
#[inline]
pub fn soft_clip(x: f32) -> f32 {
    tanhf(x)
}

/// Hard clip to ±1 range.
///
/// Abrupt limiting that creates flat tops on waveforms.
/// Produces harsh odd harmonics.
#[inline]
pub fn hard_clip(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

/// Foldback distortion.
///
/// When |x| exceeds threshold, the signal "folds" back.
/// Creates rich harmonic content, popular in synthesizers.
#[inline]
pub fn foldback(x: f32, threshold: f32) -> f32 {
    if fabsf(x) <= threshold {
        x
    } else {
        // Fold the signal back
        let sign = if x > 0.0 { 1.0 } else { -1.0 };
        let excess = fabsf(x) - threshold;
        let folded = threshold - excess;
        // Recursive fold for high drive
        if folded < -threshold {
            foldback(sign * folded, threshold)
        } else {
            sign * folded
        }
    }
}

/// Asymmetric soft clipping.
///
/// Positive and negative halves clip differently, producing
/// both even and odd harmonics (warmer, tube-like character).
#[inline]
pub fn asymmetric_clip(x: f32) -> f32 {
    if x >= 0.0 {
        // Positive: gentler clipping
        tanhf(x)
    } else {
        // Negative: harder clipping (reaches limit faster)
        tanhf(x * 1.5) / 1.5 * 1.2
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Convert decibels to linear gain.
#[inline]
pub fn db_to_linear(db: f32) -> f32 {
    // 10^(dB/20) = e^(dB * ln(10)/20)
    const FACTOR: f32 = core::f32::consts::LN_10 / 20.0;
    expf(db * FACTOR)
}

/// Convert linear gain to decibels.
#[inline]
pub fn linear_to_db(linear: f32) -> f32 {
    // 20 * log10(linear) = 20 * ln(linear) / ln(10)
    const FACTOR: f32 = 20.0 / core::f32::consts::LN_10;
    #[cfg(feature = "std")]
    {
        linear.ln() * FACTOR
    }
    #[cfg(not(feature = "std"))]
    {
        libm::logf(linear) * FACTOR
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 48000.0;

    #[test]
    fn soft_clip_bounds() {
        // Soft clip should approach but never exceed ±1
        // Note: tanh(100.0) = 1.0 exactly in f32 due to saturation
        // Use smaller values where tanh is measurably < 1.0
        assert!(soft_clip(3.0) < 1.0);
        assert!(soft_clip(3.0) > 0.99); // tanh(3) ≈ 0.9951
        assert!(soft_clip(-3.0) > -1.0);
        assert!(soft_clip(-3.0) < -0.99);
        // At extreme values, it saturates to exactly ±1.0 (acceptable)
        assert!(soft_clip(100.0) <= 1.0);
        assert!(soft_clip(-100.0) >= -1.0);
    }

    #[test]
    fn soft_clip_passes_zero() {
        assert!((soft_clip(0.0)).abs() < 1e-6);
    }

    #[test]
    fn hard_clip_bounds() {
        assert_eq!(hard_clip(2.0), 1.0);
        assert_eq!(hard_clip(-2.0), -1.0);
        assert_eq!(hard_clip(0.5), 0.5);
    }

    #[test]
    fn foldback_basic() {
        let threshold = 0.8;
        // Below threshold: unchanged
        assert!((foldback(0.5, threshold) - 0.5).abs() < 1e-6);
        // At threshold: unchanged
        assert!((foldback(0.8, threshold) - 0.8).abs() < 1e-6);
        // Above threshold: folds back
        let folded = foldback(1.0, threshold);
        assert!(
            (folded - 0.6).abs() < 1e-6,
            "Expected 0.6, got {}",
            folded
        );
    }

    #[test]
    fn asymmetric_clip_asymmetry() {
        // Positive and negative should behave differently
        let pos = asymmetric_clip(2.0);
        let neg = asymmetric_clip(-2.0);
        // Both should be bounded
        assert!(pos < 1.0 && pos > 0.9);
        assert!(neg > -1.0 && neg < -0.7);
        // They should NOT be symmetric
        assert!((pos + neg).abs() > 0.01, "Should be asymmetric");
    }

    #[test]
    fn db_conversion_roundtrip() {
        let original = 0.5;
        let db = linear_to_db(original);
        let back = db_to_linear(db);
        assert!(
            (original - back).abs() < 1e-5,
            "Roundtrip failed: {} -> {} -> {}",
            original,
            db,
            back
        );
    }

    #[test]
    fn db_conversion_known_values() {
        // 0 dB = 1.0 linear
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-6);
        // -6 dB ≈ 0.5 linear
        assert!((db_to_linear(-6.0206) - 0.5).abs() < 0.001);
        // +6 dB ≈ 2.0 linear
        assert!((db_to_linear(6.0206) - 2.0).abs() < 0.001);
    }

    #[test]
    fn distortion_unity_gain() {
        let mut dist = Distortion::new(SAMPLE_RATE);
        dist.set_drive_db(0.0);
        dist.set_level_db(0.0);
        dist.set_tone_hz(20000.0); // High cutoff = nearly bypass
        dist.reset(); // Snap params to target

        // Small signal should pass through nearly unchanged
        let input = 0.1;
        // Run a few samples to stabilize filter
        for _ in 0..100 {
            dist.process(input);
        }
        let output = dist.process(input);

        // Should be close to input (soft_clip(0.1) ≈ 0.0997, filter adds tiny lag)
        assert!(
            (output - input).abs() < 0.02,
            "Expected ~{}, got {}",
            input,
            output
        );
    }

    #[test]
    fn distortion_increases_with_drive() {
        let mut dist = Distortion::new(SAMPLE_RATE);
        dist.set_tone_hz(20000.0);
        dist.reset();

        let input = 0.1;

        // Low drive
        dist.set_drive_db(0.0);
        dist.reset();
        for _ in 0..100 {
            dist.process(input);
        }
        let low_drive_output = dist.process(input).abs();

        // High drive
        dist.set_drive_db(20.0);
        dist.reset();
        for _ in 0..100 {
            dist.process(input);
        }
        let high_drive_output = dist.process(input).abs();

        // Higher drive should produce higher output (saturated)
        assert!(
            high_drive_output > low_drive_output,
            "High drive {} should exceed low drive {}",
            high_drive_output,
            low_drive_output
        );
    }

    #[test]
    fn distortion_tone_affects_brightness() {
        let mut dist = Distortion::new(SAMPLE_RATE);
        dist.set_drive_db(20.0);
        dist.set_level_db(0.0);

        // Generate a simple impulse response to measure filter effect
        // Dark tone
        dist.set_tone_hz(1000.0);
        dist.reset();
        let _dark = dist.process(1.0);
        let dark_decay = dist.process(0.0).abs();

        // Bright tone
        dist.set_tone_hz(10000.0);
        dist.reset();
        let _bright = dist.process(1.0);
        let bright_decay = dist.process(0.0).abs();

        // Lower cutoff should have slower response (more filtering)
        assert!(
            dark_decay < bright_decay,
            "Dark {} should decay slower than bright {}",
            dark_decay,
            bright_decay
        );
    }
}
