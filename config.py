SR =  44100
window_size = 1024
hop_length = 768

patch_size = 128 # roughly 33 seconds

## for training
EPOCH = 10000
BATCH = 128
SAMPLING_STRIDE = 10

## for validation
EARLY_STOPPING_PATIENCE = 15