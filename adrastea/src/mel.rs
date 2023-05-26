/*
 * This file is part of Adrastea.
 *
 * Adrastea is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Affero General Public License as published by the Free Software
 * Foundation, version 3.
 *
 * Adrastea is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along
 * with Adrastea. If not, see <https://www.gnu.org/licenses/>.
 */

use core::f32::consts::PI;

use rustfft::num_complex::Complex32;

use crate::stft::RealStft;

const MEL_F_STEP: f32 = 200.0 / 3.0;
const MEL_LOG_STEP: f32 = 0.06875177742094912;
const MEL_LOG_THRESHOLD: f32 = 15.0;
const MEL_HZ_THRESHOLD: f32 = 1000.0;

// librosa-compatible mel transform
pub fn hz_to_mel(hz: f32) -> f32 {
    if hz >= MEL_HZ_THRESHOLD {
        MEL_LOG_THRESHOLD + (hz / MEL_HZ_THRESHOLD).ln() / MEL_LOG_STEP
    } else {
        hz / MEL_F_STEP
    }
}

pub fn mel_to_hz(mel: f32) -> f32 {
    if mel >= MEL_LOG_THRESHOLD {
        MEL_HZ_THRESHOLD * ((mel - MEL_LOG_THRESHOLD) * MEL_LOG_STEP).exp()
    } else {
        MEL_F_STEP * mel
    }
}

pub fn mel_frequencies(out: &mut [f32], fmax: f32) {
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(fmax);
    let mel_step = (mel_max - mel_min) / (out.len() as f32 - 1.0);
    for i in 0..out.len() {
        out[i] = mel_to_hz(mel_min + mel_step * (i as f32));
    }
}

pub fn fft_frequencies(out: &mut [f32], n_fft: usize, nyquist: f32) {
    let n_bins = n_fft / 2;
    let df = nyquist / (n_bins as f32);
    for idx in 0..out.len() {
        out[idx] = df * (idx as f32);
    }
}

pub fn mel_filter_bank(out: &mut [f32], n_mels: usize, n_fft: usize, sample_rate: f32) {
    let mut mel_freqs = vec![0.0; n_mels + 2];
    let mut fft_freqs = vec![0.0; n_fft / 2 + 1];
    let nyquist = sample_rate / 2.0;
    mel_frequencies(&mut mel_freqs, nyquist);
    fft_frequencies(&mut fft_freqs, n_fft, nyquist);
    for y in 0..n_mels {
        let (m_lo, m_mid, m_hi) = (mel_freqs[y], mel_freqs[y + 1], mel_freqs[y + 2]);
        for x in 0..fft_freqs.len() {
            let lo = (fft_freqs[x] - m_lo) / (m_mid - m_lo);
            let hi = (m_hi - fft_freqs[x]) / (m_hi - m_mid);
            let weight = 0.0f32.max(lo.min(hi));
            // slaney norm
            out[y * fft_freqs.len() + x] = weight * 2.0 / (m_hi - m_lo);
        }
    }
}

fn hann_window(n: usize) -> Vec<f32> {
    let mut window = vec![0.0; n];
    for i in 0..n {
        window[i] = 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos());
    }
    window
}

pub struct LogMelSpectrogramTransform {
    filter_bank: Vec<f32>,
    stft_plan: RealStft,
    n_mels: usize,
    n_fft: usize,
}

impl LogMelSpectrogramTransform {
    pub fn new(n_fft: usize, n_mels: usize, hop_length: usize, sample_rate: f32) -> Self {
        let n_bins = n_fft / 2 + 1;
        let mut filter_bank = vec![0.0; n_mels * n_bins];
        mel_filter_bank(&mut filter_bank, n_mels, n_fft, sample_rate);
        let stft_plan = RealStft::new(n_fft, hop_length, hann_window(n_fft));
        Self {
            filter_bank,
            stft_plan,
            n_mels,
            n_fft,
        }
    }

    pub fn complex_scratch_size(&self, num_samples: usize) -> usize {
        self.stft_plan.output_size(num_samples)
    }

    pub fn real_scratch_size(&self, num_samples: usize) -> usize {
        self.num_cols(num_samples) * self.stft_plan.num_bins()
    }

    pub fn num_cols(&self, num_samples: usize) -> usize {
        self.stft_plan.num_frames(num_samples) - 1
    }

    pub fn output_size(&self, num_samples: usize) -> usize {
        self.num_cols(num_samples) * self.n_mels
    }

    pub fn process(
        &mut self,
        mel_spec: &mut [f32],
        samples: &[f32],
        complex_scratch: &mut [Complex32],
        real_scratch: &mut [f32],
    ) {
        let (stft_c, stft_mag) = (complex_scratch, real_scratch);
        let n_frames = self.stft_plan.num_frames(samples.len()) - 1;
        self.stft_plan.process(stft_c, &samples);
        for y in 0..self.stft_plan.num_bins() {
            for x in 0..n_frames {
                let norm = stft_c[y * (n_frames + 1) + x].norm();
                stft_mag[y * n_frames + x] = norm * norm;
            }
        }
        let (m, k, n) = (self.n_mels, self.n_fft / 2 + 1, n_frames);
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                self.filter_bank.as_ptr(),
                k as isize,
                1,
                stft_mag.as_ptr(),
                n as isize,
                1,
                0.0,
                mel_spec.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        let mut log_max = 1.0e-10f32;
        for v in mel_spec.iter_mut() {
            *v = v.max(1.0e-10).log10();
            log_max = log_max.max(*v);
        }
        for v in mel_spec {
            *v = (v.max(log_max - 8.0) + 4.0) / 4.0;
        }
    }
}
