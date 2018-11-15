import tensorflow as tf

from U_net import U_Net
from util import *
from config import *
from sklearn.model_selection import ShuffleSplit

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def train_val(path_spectro_macbook, path_spectro_huawei_sla_al00):
    
    X_list_macbook, Y_list_macbook = [], []
    X_list_huawei_sla_al00, Y_list_huawei_sla_al00 = [], []
    # three-level of folder structure
    folders_1st = os.listdir(path_spectro_macbook)
    for f_1st in folders_1st:
        if "DS_Store" not in f_1st:
            folders_2nd = os.listdir(os.path.join(path_spectro_macbook, f_1st))
            for f_2nd in folders_2nd:
                if "DS_Store" not in f_2nd:
                    folders_3rd = os.listdir(os.path.join(path_spectro_macbook, f_1st, f_2nd))
                    for f_3rd in folders_3rd:
                        if "DS_Store" not in f_3rd:
                            path_3rd_macbook = os.path.join(path_spectro_macbook, f_1st, f_2nd, f_3rd)
                            X_list_macbook_temp, Y_list_macbook_temp = LoadSpectrogram(path_3rd_macbook)
                            X_list_macbook += X_list_macbook_temp
                            Y_list_macbook += Y_list_macbook_temp
                            path_3rd_huawei_sla_al00 = os.path.join(path_spectro_huawei_sla_al00, f_1st, f_2nd, f_3rd)
                            X_list_huawei_sla_al00_temp, Y_list_huawei_sla_al00_temp = LoadSpectrogram(path_3rd_huawei_sla_al00)
                            X_list_huawei_sla_al00 += X_list_huawei_sla_al00_temp
                            Y_list_huawei_sla_al00 += Y_list_huawei_sla_al00_temp
                                    
    # macbook spectrogram
    X_mag_macbook, X_phase_macbook = Magnitude_phase(X_list_macbook)
    Y_mag_macbook, _ = Magnitude_phase(Y_list_macbook)

    # huawei mobile phone spectrogram
    X_mag_huawei_sla_al00, X_phase_huawei_sla_al00 = Magnitude_phase(X_list_huawei_sla_al00)
    Y_mag_huawei_sla_al00, _ = Magnitude_phase(Y_list_huawei_sla_al00)

    X_mag = X_mag_macbook + X_mag_huawei_sla_al00
    Y_mag = Y_mag_macbook + Y_mag_huawei_sla_al00
    X_phase = X_phase_macbook + X_phase_huawei_sla_al00

    rs = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0, train_size=None)
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

        # for ii in range(len(X_train)):
        #     f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        #     ax1.imshow(X_train[ii].reshape((512, 128)), origin='lower')
        #     ax1.set_title('IR convolved spectrogram')
        #     ax2.imshow(y_train[ii].reshape((512, 128)), origin='lower')
        #     ax2.set_title('Clean tone spectrogram')
        #     plt.show()
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
    path_spectro_macbook = "/home/rong/de-artefact_data/VocalSet/macbook_reverb_spectro"
    path_spectro_huawei_sla_al00 = "/home/rong/de-artefact_data/VocalSet/huawei_sla_al00_reverb_spectro"
    train_val(path_spectro_macbook, path_spectro_huawei_sla_al00) 
    print("Training Complete!!")