class DDAE_Hyperparams:
    drop_rate = 0.5
    n_units = 1024
    activation = 'lrelu'
    final_act = 'linear'
    f_bin = 257

class CBHG_Hyperparams:
    #### Modules ####
    drop_rate = 0.0
    normalization_mode = 'layer_norm'
    activation = 'lrelu'
    final_act = 'linear'
    f_bin = 257
    ## CBHG encoder
    banks_filter = 64
    n_banks = 8
    n_resblocks = 3
    resblocks_filter = 512
    down_sample = [512]
    n_highway = 4
    gru_size = 256

class UNET_Hyperparams:
    drop_rate = 0.0
    normalization_mode = 'batch_norm'
    activation = 'lrelu'
    final_act = 'linear'
    f_bin = 257
    enc_layers = [32,64,64,128,128,256,256,512,512,1024]
    dec_layers = [1024,512,512,256,256,128,128,64,64,32]


class Hyperparams:    
    is_variable_length = False
    #### Signal Processing ####
    n_fft = 512
    hop_length = 256
    n_mels = 80 # Number of Mel banks to generate
    f_bin = n_fft//2 + 1
    ppg_dim = 96 
    feature_type = 'logmag' # logmag lps
    nfeature_mode = 'None' # mean_std minmax
    cfeature_mode = 'None' # mean_std minmax

    # num_layers = [1024,512,256,128,256,512] # bigCNN
    num_layers = [2048,1024,512,256,128,256,512] # bigCNN2
    normalization_mode = 'None'
    activation = 'lrelu'
    final_act = 'relu'

    n_units = 1024

    pretrain = True
    pretrain_path = "/mnt/Data/user_vol_2/user_neillu/DNS_Challenge_timit/models/20200516_timit_broad_confusion_triphone_newts/model-1000000"

    SAVEITER = 20000
    # is_trimming = True
    # sr = 16000 # Sample rate.
    # n_fft = 1024 # fft points (samples)
    # frame_shift = 0.0125 # seconds
    # frame_length = 0.05 # seconds
    # hop_length = int(sr*frame_shift) # samples.
    # win_length = int(sr*frame_length) # samples.
    # power = 1.2 # Exponent for amplifying the predicted magnitude
    # n_iter = 200 # Number of inversion iterations
    # preemphasis = .97 # or None
    max_db = 100
    ref_db = 20
