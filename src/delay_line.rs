//! Delay line with linear interpolation for time-based effects.
//!
//! Provides a circular buffer with fractional sample delay support,
//! essential for chorus, flanger, and vibrato effects.

/// Interpolated delay line using a circular buffer.
///
/// Supports fractional delay times through linear interpolation,
/// allowing smooth modulation of delay time without artifacts.
///
/// # Memory
///
/// The buffer is heap-allocated during construction but never reallocates.
/// No allocations occur during audio processing.
///
/// # Example
///
/// ```
/// use rust_dsp_effects::InterpolatedDelay;
///
/// // 50ms max delay at 44.1kHz
/// let max_delay_samples = (0.05 * 44100.0) as usize;
/// let mut delay = InterpolatedDelay::new(max_delay_samples);
///
/// // Read with 10.5 sample delay (fractional)
/// let output = delay.read(10.5);
/// delay.write(1.0);
/// ```
#[derive(Debug, Clone)]
pub struct InterpolatedDelay {
    /// Circular buffer storage
    buffer: alloc::vec::Vec<f32>,
    /// Write position in buffer
    write_pos: usize,
}

// For no_std support
#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec;

#[cfg(feature = "std")]
extern crate std as alloc;

impl InterpolatedDelay {
    /// Creates a new delay line with the given maximum delay in samples.
    ///
    /// # Arguments
    ///
    /// * `max_delay_samples` - Maximum delay capacity in samples
    ///
    /// # Panics
    ///
    /// Panics if `max_delay_samples` is 0.
    pub fn new(max_delay_samples: usize) -> Self {
        assert!(max_delay_samples > 0, "Delay size must be > 0");

        Self {
            buffer: vec![0.0; max_delay_samples],
            write_pos: 0,
        }
    }

    /// Reads a delayed sample with linear interpolation.
    ///
    /// # Arguments
    ///
    /// * `delay_samples` - Delay time in samples (can be fractional)
    ///
    /// Returns the interpolated sample from the delay line.
    ///
    /// # Panics
    ///
    /// Panics if `delay_samples` is negative or exceeds the buffer capacity.
    #[inline]
    pub fn read(&self, delay_samples: f32) -> f32 {
        debug_assert!(delay_samples >= 0.0);
        debug_assert!(delay_samples < self.buffer.len() as f32);

        // Calculate read position (going backwards from last written sample)
        // write_pos points to where the NEXT sample will be written,
        // so the last written sample is at write_pos - 1
        let delay_int = delay_samples as usize;
        let delay_frac = delay_samples - delay_int as f32;

        let buffer_len = self.buffer.len();

        // Calculate position of last written sample
        let last_written = if self.write_pos == 0 {
            buffer_len - 1
        } else {
            self.write_pos - 1
        };

        // Read position (going back from last written)
        let read_pos = if last_written >= delay_int {
            last_written - delay_int
        } else {
            buffer_len + last_written - delay_int
        };

        // Next sample position for interpolation (one sample further back)
        let next_pos = if read_pos == 0 {
            buffer_len - 1
        } else {
            read_pos - 1
        };

        // Linear interpolation: y = y0 + (y1 - y0) * frac
        let sample0 = self.buffer[read_pos];
        let sample1 = self.buffer[next_pos];

        sample0 + (sample1 - sample0) * delay_frac
    }

    /// Writes a sample to the delay line and advances the write position.
    ///
    /// # Arguments
    ///
    /// * `sample` - The sample to write
    #[inline]
    pub fn write(&mut self, sample: f32) {
        self.buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
    }

    /// Clears the delay line (sets all samples to 0).
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }

    /// Returns the maximum delay capacity in samples.
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_basic() {
        let mut delay = InterpolatedDelay::new(10);

        // Write some samples
        for i in 1..=5 {
            delay.write(i as f32);
        }

        // Read with 3 sample delay (should get value from 2 samples ago)
        delay.write(6.0);
        let output = delay.read(3.0);
        assert_eq!(output, 3.0);
    }

    #[test]
    fn test_delay_interpolation() {
        let mut delay = InterpolatedDelay::new(10);

        // Write 0, 1, 2, 3
        delay.write(0.0);
        delay.write(1.0);
        delay.write(2.0);
        delay.write(3.0);

        // Read with 1.5 sample delay
        // Should interpolate between sample at delay=1 (value 2.0) and delay=2 (value 1.0)
        // Result: 2.0 + (1.0 - 2.0) * 0.5 = 2.0 - 0.5 = 1.5
        let output = delay.read(1.5);
        assert!((output - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_delay_wrap_around() {
        let mut delay = InterpolatedDelay::new(4);

        // Fill buffer completely
        delay.write(1.0);
        delay.write(2.0);
        delay.write(3.0);
        delay.write(4.0);

        // Now write_pos wraps to 0
        delay.write(5.0); // write_pos = 0, buffer = [5, 2, 3, 4]

        // Read with delay that crosses the wrap boundary
        let output = delay.read(3.0);
        assert_eq!(output, 2.0);
    }

    #[test]
    fn test_delay_clear() {
        let mut delay = InterpolatedDelay::new(5);

        delay.write(10.0);
        delay.write(20.0);

        delay.clear();

        let output = delay.read(1.0);
        assert_eq!(output, 0.0);
        assert_eq!(delay.write_pos, 0);
    }

    #[test]
    #[should_panic]
    fn test_delay_zero_size_panics() {
        let _delay = InterpolatedDelay::new(0);
    }
}
