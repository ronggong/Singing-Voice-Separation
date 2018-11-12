import tensorflow as tf
import librosa
from librosa.util import find_files
from librosa import load

from util import *
from config import *
from U_net import U_Net


def main() :
    music_path = "../test_test_dataset/f3_scales_c_slow_forte_a_reverb.wav"
    
    input_wav_mag, input_wav_phase = LoadAudio(music_path)

    pad_size = patch_size - input_wav_mag.shape[1] % patch_size

    input_wav_mag = np.hstack((input_wav_mag, np.zeros((513, pad_size), dtype=np.float32)))

    output_mask = np.zeros((512, input_wav_mag.shape[1]))

    for ii in range(int(input_wav_mag.shape[1]/patch_size)):
        START = patch_size * ii
        END = START + patch_size  # 11 seconds

        X=input_wav_mag[1:, START:END].reshape(1,512,128,1)

        # input_wav_phase=input_wav_phase[:, START:END]
        # X = input_wav_mag[1:].reshape(1,512,128,1)
        
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(x = {"mag":X},y = None, num_epochs = 1,shuffle = False)
        
        deep_u_net = tf.estimator.Estimator(model_fn=U_Net, model_dir="./model")
        predictions = list(deep_u_net.predict(input_fn=predict_input_fn))
        mask = predictions[0]['outputs']
        mask = mask.reshape(512, patch_size)
        output_mask[:, START:END] = mask
    
    target_pred_mag = np.vstack((np.zeros((output_mask.shape[1])), output_mask))
    target_pred_mag = target_pred_mag[:, :target_pred_mag.shape[1]-pad_size]
    # target_pred_mag = np.vstack((np.zeros((128)), mask.reshape(512, 128)))
    SaveAudio(music_path[:-4]+"_output.wav",target_pred_mag, input_wav_phase)

if __name__ == "__main__" : 
    main()