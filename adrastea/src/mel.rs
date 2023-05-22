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

pub fn mel_filter_bank(out: &mut [f64], n_mels: i32, n_fft: i32, sample_rate: f64) {
    todo!()
}
