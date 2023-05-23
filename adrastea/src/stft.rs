use alloc::sync::Arc;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};

pub struct RealStft {
    n_fft: usize,
    hop_length: usize,
    window_fn: Vec<f32>,
    frame: Vec<Complex32>,
    fft: Vec<Complex32>,
    scratch: Vec<Complex32>,
    plan: Arc<dyn Fft<f32>>,
}

impl RealStft {
    pub fn new(n_fft: usize, hop_length: usize, window_fn: Vec<f32>) -> Self {
        let plan = FftPlanner::new().plan_fft_forward(n_fft);
        let frame = vec![Complex32::new(0.0, 0.0); n_fft];
        let fft = vec![Complex32::new(0.0, 0.0); n_fft];
        let scratch = vec![Complex32::new(0.0, 0.0); plan.get_outofplace_scratch_len()];
        Self {
            n_fft,
            hop_length,
            window_fn,
            plan,
            frame,
            fft,
            scratch,
        }
    }

    pub fn output_size(&self, input_size: usize) -> usize {
        (1 + input_size / self.hop_length) * (self.n_fft / 2 + 1)
    }

    pub fn process(&mut self, output: &mut [Complex32], wave: &[f32]) {
        let n_frames = 1 + wave.len() / self.hop_length;
        for i in 0..n_frames {
            let center = i * self.hop_length;
            let pad = self.n_fft / 2;
            for j in 0..self.n_fft {
                let idx = center as isize + j as isize - pad as isize;
                // mirror when out of bounds
                self.frame[j] = match (idx >= 0, idx < wave.len() as isize) {
                    (true, true) => Complex32::new(wave[idx as usize], 0.0) * self.window_fn[j],
                    (true, false) => {
                        Complex32::new(
                            wave[wave.len() - 1 - (idx - wave.len() as isize) as usize],
                            0.0,
                        ) * self.window_fn[j]
                    }
                    (false, true) => Complex32::new(wave[-idx as usize], 0.0) * self.window_fn[j],
                    (false, false) => unreachable!(),
                }
            }
            self.plan.process_outofplace_with_scratch(
                &mut self.frame,
                &mut self.fft,
                &mut self.scratch,
            );
            for j in 0..self.n_fft / 2 + 1 {
                output[j * n_frames + i] = self.fft[j];
            }
        }
    }
}
