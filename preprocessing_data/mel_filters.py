"""
Mel filter bank generation for TensorFlow - matching OpenAI Whisper exactly
Uses librosa-compatible mel filter generation
"""

import os
# Fix TensorFlow memory issues BEFORE importing TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np
from audio_constants import SAMPLE_RATE, N_FFT, N_MELS


def mel_frequencies(n_mels, fmin=0.0, fmax=None, htk=False):
    """Mel frequencies matching librosa implementation"""
    if fmax is None:
        fmax = float(SAMPLE_RATE) / 2
    
    if htk:
        # HTK mel scale
        min_mel = 2595.0 * np.log10(1.0 + fmin / 700.0)
        max_mel = 2595.0 * np.log10(1.0 + fmax / 700.0)
    else:
        # Slaney mel scale (librosa default)
        f_min = 0.0
        f_sp = 200.0 / 3
        
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0
        
        if fmin >= min_log_hz:
            min_mel = min_log_mel + np.log(fmin / min_log_hz) / logstep
        else:
            min_mel = (fmin - f_min) / f_sp
            
        if fmax >= min_log_hz:
            max_mel = min_log_mel + np.log(fmax / min_log_hz) / logstep
        else:
            max_mel = (fmax - f_min) / f_sp
    
    # Linear space in mel scale
    mels = np.linspace(min_mel, max_mel, n_mels)
    
    if htk:
        # HTK inverse mel scale
        return 700.0 * (10**(mels / 2595.0) - 1.0)
    else:
        # Slaney inverse mel scale
        freqs = np.empty_like(mels)
        
        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels
        
        # Fill in the log scale
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0
        
        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
        
        return freqs


def mel_filter_bank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=None, norm="slaney"):
    """
    Create mel filter bank exactly matching librosa.filters.mel
    This matches the filters used in OpenAI Whisper mel_filters.npz
    """
    if fmax is None:
        fmax = float(sr) / 2
    
    # Get mel frequencies
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=False)
    
    # Convert to FFT bin indices
    fftfreqs = np.linspace(0, float(sr) / 2, int(1 + n_fft // 2), endpoint=True)
    mel_f = np.asarray(mel_f)
    
    # Create filter bank
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)
    
    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
    
    for i in range(n_mels):
        # Lower and upper slopes for each mel band
        lower = (fftfreqs - mel_f[i]) / (mel_f[i+1] - mel_f[i])
        upper = (mel_f[i+2] - fftfreqs) / (mel_f[i+2] - mel_f[i+1])
        
        # Intersection of the two slopes
        weights[i] = np.maximum(0, np.minimum(lower, upper))
        
        if norm == "slaney":
            weights[i] *= enorm[i]
    
    return weights


def get_mel_filters():
    """Get mel filter bank matching OpenAI Whisper exactly"""
    # OpenAI removes last frequency bin, so we need filters for N_FFT//2 bins, not N_FFT//2 + 1
    filters = mel_filter_bank(SAMPLE_RATE, N_FFT, N_MELS)
    
    # Remove last column to match OpenAI stft[..., :-1] behavior
    # Original shape: (n_mels, n_fft//2 + 1), after remove: (n_mels, n_fft//2)
    filters = filters[:, :-1]  # Shape: (80, 200) for N_FFT=400
    
    return tf.constant(filters, dtype=tf.float32)
