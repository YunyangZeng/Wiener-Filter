# -*- coding: utf-8 -*-
"""
@author: yunyang zeng
"""
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal 
def wiener(noisy, sr = 44100): 
    mu = 0.98 #smoothing factor in noise spectrum update
    a_dd = 0.98 #smoothing factor in priori update
    eta = 0.15 #VAD threshold
    frame_dur = 0.02 # frame duration, 20ms hamming window
    frame_length = int(frame_dur * sr) # frame length in samples
    hamming_window = np.hamming(frame_length) # 20ms hamming window
    U = sum(hamming_window**2)/frame_length # normalization constant for welch method

    len_first_120ms = int(sr * 0.12)
    first_120ms = noisy[0:len_first_120ms]
    
    '''Estimate noise power spectral density using Welch Method'''
    number_of_frames_first_120ms = len_first_120ms//(frame_length//2)-1
    noise_psd = np.zeros([frame_length, ])
    n_start = 0
    
    for i in range(number_of_frames_first_120ms):
        noise = first_120ms[n_start : n_start + frame_length]
        noise = noise * hamming_window
        noise_fft = fft.fft(noise, n=frame_length)
        noise_psd += (np.abs(noise_fft) ** 2) / (frame_length * U)
        n_start += int(frame_length / 2)
    noise_psd /= number_of_frames_first_120ms
    
    
    n_noisy_frames = len(noisy)//(frame_length//2) - 1
    n_start = 0
    enhanced_signal = np.zeros(noisy.shape)
    for j in range(n_noisy_frames):
        noisy_frame = noisy[n_start : n_start + frame_length] * hamming_window
        noisy_frame_fft = fft.fft(noisy_frame, n=frame_length)
        noisy_psd = (np.abs(noisy_frame_fft) ** 2) / (frame_length * U)
        '''========================VAD====================='''
        if j == 0:
            posterior_SNR = noisy_psd / noise_psd
            posterior_SNR_prime = posterior_SNR - 1
            posterior_SNR_prime = np.maximum(0, posterior_SNR_prime)
            priori_SNR = a_dd + (1 - a_dd) * posterior_SNR_prime
            G = np.real((priori_SNR / (priori_SNR + 1)) ** 0.5)
            posterior_SNR_prev = posterior_SNR
        else:
            G_prev = G
            posterior_SNR = noisy_psd / noise_psd
            posterior_SNR_prime = posterior_SNR -1
            posterior_SNR_prime = np.maximum(0, posterior_SNR_prime)
            priori_SNR = a_dd * (G_prev ** 2) * posterior_SNR_prev + (1 - a_dd) *posterior_SNR_prime
            posterior_SNR_prev = posterior_SNR
            G = np.real((priori_SNR / (priori_SNR + 1)) ** 0.5)
        log_sigma_k = posterior_SNR * priori_SNR /(1 + priori_SNR) - np.log(1 + priori_SNR)
        vad_decision = np.real(sum(log_sigma_k) / frame_length)
        if vad_decision < eta:
            noise_psd =mu * noise_psd + (1 - mu) * noisy_psd
        '''===================end of VAD ==================='''
        enhanced_frame = np.real(fft.ifft(noisy_frame_fft * G, n = frame_length))
        if j == 0:
            enhanced_signal[n_start : n_start + frame_length//2 ] = enhanced_frame[0 : frame_length//2 ]
            overlap = enhanced_frame[frame_length//2 : frame_length ]
        else:
            enhanced_signal[n_start : n_start + frame_length//2 ] = overlap + enhanced_frame[0 : frame_length//2 ]
            overlap = enhanced_frame[frame_length//2 : frame_length ]
        n_start += frame_length // 2
    enhanced_signal[n_start : n_start + frame_length // 2  ] = overlap
    
    return enhanced_signal

noisy, sr = sf.read("./noisy.wav")
enhanced = wiener(noisy,sr)
sf.write("./wiener_enhanced.wav", enhanced , sr)

