import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

from U_net import U_Net
from util import *
from config import *
from sklearn.model_selection import ShuffleSplit


def train(path_spectro) : 
    
    X_list, Y_list = LoadSpectrogram(path_spectro) # Mix spectrogram
    X_mag, X_phase = Magnitude_phase(X_list)
    Y_mag,_ = Magnitude_phase(Y_list)
    deep_u_net = tf.estimator.Estimator(model_fn=U_Net, model_dir="./model")
    
    for e in range(EPOCH) :
        # Random sampling for training
        X, y = sampling(X_mag, Y_mag)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x = {"mag": X}, y = y, batch_size = BATCH, num_epochs = 1, shuffle = False)
    
        deep_u_net.train(input_fn = train_input_fn)


def train_val_tone(path_spectro_macbook_tone, path_spectro_huawei_sla_al00_tone) : 
    # macbook spectrogram
    X_list_macbook, Y_list_macbook = LoadSpectrogram(path_spectro_macbook_tone)
    X_mag_macbook, X_phase_macbook = Magnitude_phase(X_list_macbook)
    Y_mag_macbook, _ = Magnitude_phase(Y_list_macbook)

    # huawei mobile phone spectrogram
    X_list_huawei_sla_al00, Y_list_huawei_sla_al00 = LoadSpectrogram(path_spectro_huawei_sla_al00_tone)
    X_mag_huawei_sla_al00, X_phase_huawei_sla_al00 = Magnitude_phase(X_list_huawei_sla_al00)
    Y_mag_huawei_sla_al00, _ = Magnitude_phase(Y_list_huawei_sla_al00)

    X_mag = X_mag_macbook + X_mag_huawei_sla_al00
    Y_mag = Y_mag_huawei_sla_al00 + Y_mag_huawei_sla_al00
    X_phase = X_phase_macbook + X_phase_huawei_sla_al00

    rs = ShuffleSplit(n_splits=1, test_size=0.25, random_state=0, train_size=None)
    for train_index, test_index in rs.split(X_mag):
        X_mag_train = [X_mag[ii] for ii in train_index]
        X_mag_val = [X_mag[ii] for ii in test_index]

        Y_mag_train = [Y_mag[ii] for ii in train_index]
        Y_mag_val = [Y_mag[ii] for ii in test_index]

        X_phase_train = [X_phase[ii] for ii in train_index]
        X_phase_test = [X_phase[ii] for ii in test_index]

    deep_u_net = tf.estimator.Estimator(model_fn=U_Net, model_dir="./model")
    
    es_counter = 0 # early stopping counter
    min_val_loss = 10000
    for e in range(EPOCH) :
        # Random sampling for training
        X_train, y_train = sampling(X_mag_train, Y_mag_train)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x = {"mag": X_train}, y = y_train, batch_size = BATCH, num_epochs = 1, shuffle = False)
    
        # Random sampling for validation
        X_val, y_val = sampling(X_mag_val, Y_mag_val)
        val_input_fn = tf.estimator.inputs.numpy_input_fn(x = {"mag": X_val}, y = y_val, batch_size = BATCH, num_epochs = 1, shuffle = False)

        deep_u_net.train(input_fn = train_input_fn)
        val_return = deep_u_net.evaluate(input_fn = val_input_fn)
        val_loss = val_return['loss']

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= EARLY_STOPPING_PATIENCE:
                break
        
if __name__ == '__main__' :
#     path_spectro = "../test_train_dataset_spectro"
#     train(path_spectro)
    path_spectro_macbook_tone = "/home/rong/de-artefact_data/tone/macbook_reverb_spectro"
    path_spectro_huawei_sla_al00_tone = "/home/rong/de-artefact_data/tone/huawei_sla_al00_reverb_spectro"
    train_val_tone(path_spectro_macbook_tone, path_spectro_huawei_sla_al00_tone) 
    print("Training Complete!!")