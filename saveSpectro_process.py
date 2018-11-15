import librosa
from librosa.util import find_files
from librosa import load
from util import SaveSpectrogram
import os


def saveSpectro(path_dataset_input, path_dataset_output, path_spectro): 
    '''
    Save spectro for the vocal dataset
    input: IR convolved vocal dataset
    output: clean vocal dataset
    '''

    try :
        if not os.path.exists(path_spectro):
            os.makedirs(path_spectro)
        
        # three-level of folder structure
        folders_1st = os.listdir(path_dataset_output)
        for f_1st in folders_1st:
            if "DS_Store" not in f_1st:
                folders_2nd = os.listdir(os.path.join(path_dataset_output, f_1st))
                for f_2nd in folders_2nd:
                    if "DS_Store" not in f_2nd:
                        folders_3rd = os.listdir(os.path.join(path_dataset_output, f_1st, f_2nd))
                        for f_3rd in folders_3rd:
                            if "DS_Store" not in f_3rd:
                                # create processed audio folders
                                path_deep_spectro = os.path.join(path_spectro, f_1st, f_2nd, f_3rd)
                                if not os.path.exists(path_deep_spectro):
                                    os.makedirs(path_deep_spectro)
                                filenames = os.listdir(os.path.join(path_dataset_output, f_1st, f_2nd, f_3rd))
                                for f_wav in filenames:
                                    if f_wav.endswith('.wav'):
                                        fullname_wav_input = os.path.join(path_dataset_input, f_1st, f_2nd, f_3rd, f_wav)
                                        fullname_wav_output = os.path.join(path_dataset_output, f_1st, f_2nd, f_3rd, f_wav)
                                        filename_spectro = os.path.join(path_deep_spectro, f_wav+'.npz')
                                        aud_input, sr_input = load(fullname_wav_input, sr=None)
                                        aud_output, sr_output = load(fullname_wav_output, sr=None)

                                        print("Saving...")
                                        SaveSpectrogram(aud_input, aud_output, filename_spectro, sr_input, sr_output)
    except IndexError as e :
        print("Wrong Directory")
        pass

if __name__ == '__main__' :
    path_dataset_original = "/Users/jukedeckintern/Documents/de-artefact_data/VocalSet/FULL"

    path_dataset = "/Users/jukedeckintern/Documents/de-artefact_data/VocalSet/macbook_reverb"
    path_spectro_output = "/Users/jukedeckintern/Documents/de-artefact_data/VocalSet/macbook_reverb_spectro"

    saveSpectro(path_dataset, path_dataset_original, path_spectro_output)

    path_dataset = "/Users/jukedeckintern/Documents/de-artefact_data/VocalSet/huawei_sla_al00_reverb"
    path_spectro_output = "/Users/jukedeckintern/Documents/de-artefact_data/VocalSet/huawei_sla_al00_reverb_spectro"

    saveSpectro(path_dataset, path_dataset_original, path_spectro_output)
    print("Complete!!!!")
