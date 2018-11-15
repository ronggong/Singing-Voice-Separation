import librosa
from librosa.util import find_files
from librosa import load

import os, re 
import numpy as np
from config import *


def LoadAudio(file_path) :
    y, sr = load(file_path, sr=SR)
    stft = librosa.stft(y,n_fft=window_size, hop_length=hop_length)
    mag, phase = librosa.magphase(stft)
    return mag.astype(np.float32), phase

# Save Audiofile 
def SaveAudio(file_path, mag, phase) :
    y = librosa.istft(mag*phase,win_length=window_size,hop_length=hop_length)
    librosa.output.write_wav(file_path,y,SR,norm=True)
    print("Save complete!!")
    
def SaveSpectrogram(y_input, y_output, filename, orig_sr_input=44100, orig_sr_output=44100):
    if orig_sr_input != SR:
        y_input = librosa.core.resample(y_input, orig_sr_input, SR)
    if orig_sr_output != SR:
        y_output = librosa.core.resample(y_output, orig_sr_output, SR)

    S_input = np.abs(librosa.stft(y_input, n_fft=window_size, hop_length=hop_length)).astype(np.float32)
    S_output = np.abs(librosa.stft(y_output, n_fft=window_size, hop_length=hop_length)).astype(np.float32)

    norm = S_input.max()
    S_input /= norm
    S_output /= norm

    np.savez(filename, S_input=S_input, S_output=S_output)
    
def LoadSpectrogram(path_spectro) :
    filelist = find_files(path_spectro, ext="npz")
    x_list = []
    y_list = []
    for fl in filelist :
        data = np.load(fl)

        # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        # ax1.imshow(data['S_input'], origin='lower')
        # ax1.set_title('IR convolved spectrogram')
        # ax2.imshow(data['S_output'], origin='lower')
        # ax2.set_title('Clean tone spectrogram')
        # f.suptitle(fl)
        # plt.show()

        x_list.append(data['S_input'])
        y_list.append(data['S_output'])
    return x_list, y_list


def Magnitude_phase(spectrogram) :
    Magnitude_list = []
    Phase_list = []
    for X in spectrogram :
        mag, phase = librosa.magphase(X)
        Magnitude_list.append(mag)
        Phase_list.append(phase)
    return Magnitude_list, Phase_list


def sampling(X_mag, Y_mag):
    X = []
    y = []
    for x, target in zip(X_mag, Y_mag):
        starts = np.random.randint(0, x.shape[1] - patch_size, (x.shape[1] - patch_size) // SAMPLING_STRIDE)
        for start in starts:
            end = start + patch_size
            X.append(x[1:, start:end, np.newaxis])
            y.append(target[1:, start:end, np.newaxis])
    
    idx_shuffle = np.arange(len(X))
    np.random.shuffle(idx_shuffle)
    X = [X[ii] for ii in idx_shuffle]
    y = [y[ii] for ii in idx_shuffle]
    # shuffle the patch
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y