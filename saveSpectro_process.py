import librosa
from librosa.util import find_files
from librosa import load
from util import SaveSpectrogram
import os


# Save Spectrogram 
def saveSpectro(path_dataset, path_spectro) : 
    '''
    mix : original wav file
    source_1 : inst wav file 
    source_2 : vocal wac file 
    '''

    try :
        for a in os.listdir(os.path.join(path_dataset, 'input')):
            filename_audio_input = os.path.join(path_dataset, 'input', a)
            filename_audio_output = os.path.join(path_dataset, 'output', a)
            print("Song : %s" % a)

            filename_spectro = os.path.join(path_spectro, a+'.npz')
            if os.path.exists(filename_spectro) :
                print("Already exist!! Skip....")
                continue

            aud_input, sr_input = load(filename_audio_input, sr=None)
            aud_output, sr_output = load(filename_audio_output, sr=None)

            print("Saving...")
            SaveSpectrogram(aud_input, aud_output, filename_spectro, sr_input, sr_output)
    except IndexError as e :
        print("Wrong Directory")
        pass

if __name__ == '__main__' :
    path_dataset = "../test_train_dataset"
    path_spectro = "../test_train_dataset_spectro"
    saveSpectro(path_dataset, path_spectro)
    print("Complete!!!!")