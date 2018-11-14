import librosa
from librosa.util import find_files
from librosa import load
from util import SaveSpectrogram
import os


# Save Spectrogram 
def saveSpectro_test(path_dataset, path_spectro) : 
    '''
    Save spectro for the test dataset
    input : original wav file
    output : clean voice wav file 
    '''

    try :
        if not os.path.exists(path_spectro):
            os.makedirs(path_spectro)

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


def saveSpectro_tone(path_dataset_input, path_dataset_output, path_spectro) : 
    '''
    Save spectro for the tone dataset
    input: IR convolved tone
    output: clean tone
    '''

    try :
        if not os.path.exists(path_spectro):
            os.makedirs(path_spectro)

        for a in os.listdir(path_dataset_input):
            filename_audio_input = os.path.join(path_dataset_input, a)
            filename_audio_output = os.path.join(path_dataset_output, a)
            print("Calculate spectrogram for song : %s" % a)

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
    path_dataset_original = "/Users/jukedeckintern/Documents/de-artefact_data/tone/original"

    path_dataset = "/Users/jukedeckintern/Documents/de-artefact_data/tone/huawei_sla_al00_reverb"
    path_spectro_output = "/Users/jukedeckintern/Documents/de-artefact_data/tone/huawei_sla_al00_reverb_spectro"

    saveSpectro_tone(path_dataset, path_dataset_original, path_spectro_output)

    path_dataset = "/Users/jukedeckintern/Documents/de-artefact_data/tone/macbook_reverb"
    path_spectro_output = "/Users/jukedeckintern/Documents/de-artefact_data/tone/macbook_reverb_spectro"

    saveSpectro_tone(path_dataset, path_dataset_original, path_spectro_output)
    print("Complete!!!!")
