const MEL_F_STEP: f64 = 200.0 / 3.0;
const MEL_LOG_STEP: f64 = 0.06875177742094912;
const MEL_LOG_THRESHOLD: f64 = 15.0;
const MEL_HZ_THRESHOLD: f64 = 1000.0;

// librosa-compatible mel transform
pub fn hz_to_mel(hz: f64) -> f64 {
    if hz >= MEL_HZ_THRESHOLD {
        MEL_LOG_THRESHOLD + (hz / MEL_HZ_THRESHOLD).ln() / MEL_LOG_STEP
    } else {
        hz / MEL_F_STEP
    }
}

pub fn mel_to_hz(mel: f64) -> f64 {
    if mel >= MEL_LOG_THRESHOLD {
        MEL_HZ_THRESHOLD * ((mel - MEL_LOG_THRESHOLD) * MEL_LOG_STEP).exp()
    } else {
        MEL_F_STEP * mel
    }
}

pub fn mel_frequencies(out: &mut [f64], fmin: f64, fmax: f64) {
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mel_step = (mel_max - mel_min) / (out.len() as f64 - 1.0);
    for idx in 0..out.len() {
        out[idx] = mel_to_hz(mel_min + mel_step * (idx as f64));
    }
}

pub fn fft_frequencies(out: &mut [f64], n_fft: usize, nyquist: f64) {
    let n_bins = n_fft / 2;
    let df = nyquist / (n_bins as f64);
    for idx in 0..out.len() {
        out[idx] = df * (idx as f64);
    }
}

pub fn mel_filter_bank(out: &mut [f64], n_mels: usize, n_fft: usize, sample_rate: f64) {
    let mut mel_freqs = vec![0.0; n_mels + 2];
    let mut fft_freqs = vec![0.0; n_fft / 2 + 1];
    let nyquist = sample_rate / 2.0;
    mel_frequencies(&mut mel_freqs, 0.0, nyquist);
    fft_frequencies(&mut fft_freqs, n_fft, nyquist);
    for y in 0..n_mels {
        let (m_lo, m_mid, m_hi) = (mel_freqs[y], mel_freqs[y + 1], mel_freqs[y + 2]);
        for x in 0..fft_freqs.len() {
            let lo = (fft_freqs[x] - m_lo) / (m_mid - m_lo);
            let hi = (m_hi - fft_freqs[x]) / (m_hi - m_mid);
            let weight = 0.0f64.max(lo.min(hi));
            // slaney norm
            out[y * fft_freqs.len() + x] = weight * 2.0 / (m_hi - m_lo);
        }
    }
}
