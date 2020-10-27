from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import sys, os

from data_utils import *
from ops import *
from model import *
from cwgan import *


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.set_random_seed(1234)

write_tfrecord = True
use_waveform = False
batch_size = 64
learning_rate = 1e-5
iters = 1000000
frame_size = 64 # How many window in a slice
frame_shift= frame_size  # How many window shift in a slice
#Real window to time is in hop_length=hp.hop_length, win_length=hp.n_fft


mode = sys.argv[1] # stage1, stage2, test
ckpt_name = 'timit_broad_manner/'
ppg_path = "/mnt/Data/user_vol_2/user_neillu/btn_feat/broad_manner"
# ppg_path = None
log_path = '/mnt/Data/user_vol_2/user_neillu/BPCSE/logs/'+ckpt_name
model_path = '/mnt/Data/user_vol_2/user_neillu/BPCSE/models/'+ckpt_name
dev_list = "/mnt/Data/user_vol_2/user_neillu/BPCSE/dv_timit"
test_list = "/mnt/Data/user_vol_2/user_neillu/BPCSE/ts_timit_new_all"

#Add ckpt_name for different record
record_path = "/mnt/Data/user_vol_2/user_neillu/BPCSE" 

# record_name = "/log1p_None_frame{}".format(frame_size)
record_name = "/log1p_None_frame{}_broad_manner".format(frame_size)


print("log:%s" % (log_path))
    
# se = spec_CNN()
se = transformer()
# se = spec_CNNLSTM()
# se = context_DDAE()
# se = framewise_LSTM()

check_dir(log_path)
check_dir(model_path)
reader = dataPreprocessor(record_path, record_name, 
                        noisy_list="/mnt/Data/user_vol_2/user_neillu/BPCSE/tr_timit",
                        clean_path="/mnt/Corpus/TIMIT_wav_noisy/training/train_clean/",
                        ppg_path = ppg_path,
                        use_waveform=use_waveform,
                        frame_size=frame_size, 
                        shift=frame_shift)
                        # ppg_path="/mnt/Data/user_vol_2/user_neillu/BPCSE/ppg_timit_frame32/",
if write_tfrecord:
    print("Writing tfrecord...")
    reader.write_tfrecord()
clean, noisy = reader.read_and_decode(batch_size=batch_size,num_threads=32)

trainer = SE_system(    
                    se,
                    clean, noisy,
                    test_list=dev_list,
                    log_path=log_path,
                    model_path=model_path,
                    ppg_path = ppg_path,
                    use_waveform=use_waveform,
                    frame_size=frame_size,
                    lr=learning_rate
                  )
if mode == 'stage1':
    trainer.train(mode, iters)
elif mode == 'test':
    trainer.test(test_list=test_list, 
                specific_iter=int(sys.argv[2]),
                mode=mode)
