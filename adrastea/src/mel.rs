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
