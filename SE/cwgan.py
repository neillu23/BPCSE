from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import re, os
import numpy as np
import scipy.io.wavfile as wav
import librosa
import scipy
from tqdm import tqdm

from ops import *
from data_utils import *
from hyperparameters import Hyperparams as hp
# from utils import *

class SE_system(object):
    def __init__(self, 
                se,
                data_clean, data_noisy, 
                test_list,
                log_path, 
                model_path, 
                ppg=False,
                ppg_path=None,
                use_waveform=False, 
                frame_size=64,
                lr=1e-4):

        self.model_path = model_path
        self.log_path = log_path
        self.test_list = test_list
        self.use_waveform = use_waveform
        # self.ppg_data = ppg_data
        self.ppg_path = ppg_path

        self.frame_size = frame_size
        self.lr = lr
        self.se = se

        self.clean = data_clean
        self.noisy = data_noisy

        self.G_summs = []
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.config = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement = True)
        self.config.gpu_options.allocator_type = 'BFC'
        self.sess = tf.Session(config=self.config)

    def loss(self):
        # ====================
        # Build Graph
        # ====================
        enhanced = self.se(self.noisy, reuse=False, is_training=True)
        # ====================
        # Learning rate decay
        # ====================
        global_step = tf.Variable(0, trainable=False)
        ### For learning rate
        # decay_step = tf.maximum(0, (global_step - 50*1000))
        # starter_learning_rate = self.lr
        # learning_rate = tf.train.polynomial_decay(starter_learning_rate, decay_step, 50000, 0.0)
        learning_rate = self.lr #starter_learning_rate * 10**-(tf.cast(global_step, tf.float32)//100000)
        # ============================
        # Building objective functions
        # ============================   
        # self.recons_loss = tf.losses.mean_squared_error(self.clean, enhanced) 
        self.recons_loss = tf.losses.absolute_difference(self.clean, enhanced) 
        # ===================
        #  For summaries
        # ===================
        sum_l_G = []    
        sum_l_G.append(tf.summary.scalar('recons', self.recons_loss))      
        sum_l_G.append(tf.summary.image('clean', tf.transpose(self.clean, [0, 3, 2, 1]), max_outputs=6))
        sum_l_G.append(tf.summary.image('enhanced', tf.transpose(enhanced, [0, 3, 2, 1]), max_outputs=6))
        sum_l_G.append(tf.summary.image('noisy', tf.transpose(self.noisy, [0, 3, 2, 1]), max_outputs=6))
        sum_l_G.append(tf.summary.scalar('lr', learning_rate))

        ### Attention
        ### Split heads and concat at token axis
        
        if hasattr(self.se, 'attn_hists'):
            for i in range(len(self.se.attn_hists[:])):
                # attn_oh = tf.argmax(self.dec.attn_hist[i][:,:1,:,:], -1)
                # attn_oh = tf.one_hot(attn_oh, self.frame_size)
                sum_l_G.append(tf.summary.image('token_attn_{}'.format(i), 
                                                tf.transpose(self.se.attn_hists[i][:,:1,:,:], [0,2,3,1]),
                                                max_outputs=6))

        self.G_summs = [sum_l_G]
        # ===================
        # # For optimizers
        # ===================
        g_opt = None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9)\
                .minimize(self.recons_loss, var_list=self.se.vars, global_step=global_step)
            # g_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)\
            #     .minimize(self.recons_loss, var_list=self.se.vars, global_step=global_step)
        return g_opt

    def train(self, mode="stage1", iters=65000):
        if mode == "stage1":
            g_opt = self.loss()

            if tf.gfile.Exists(self.log_path+"G"):
                tf.gfile.DeleteRecursively(self.log_path+"G")
            tf.gfile.MkDir(self.log_path+"G")

            g_merged = tf.summary.merge(self.G_summs)
            G_writer = tf.summary.FileWriter(self.log_path+"G", self.sess.graph)

            self.sess.run(tf.global_variables_initializer())
            save_path = self.model_path
            print('Training:se')        

            if hp.pretrain:
                saver = tf.train.Saver(self.se.vars)
                saver.restore(self.sess, hp.pretrain_path)   
        #-----------------------------------------------------------------#
        saver = tf.train.Saver(max_to_keep=100)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=self.sess)
        try:
            while not coord.should_stop():
                for i in tqdm(range(iters)):
                    if i%100==0:
                        fetched = self.sess.run([g_merged])  
                        G_writer.add_summary(fetched[0], i)
                        # print("\rIter:{} loss:{}".format(i, fetched[0]))
                    else:
                        _ = self.sess.run([g_opt])  

                    if i % hp.SAVEITER == hp.SAVEITER-1:
                        saver.save(self.sess, save_path + 'model', global_step=i+1)
                        self.test(self.test_list, i+1, mode='valid')

                    if i == iters-1:
                        coord.request_stop()

        except tf.errors.OutOfRangeError:
            print('Done training; epoch limit')
        finally:
            coord.request_stop()
        coord.join(threads)

        return

    def test(self, test_list=None, specific_iter=None, mode='test'):
        FRAMELENGTH = self.frame_size
        OVERLAP = self.frame_size
        test_path = self.model_path

        if mode == 'test':
            reuse = False
            if specific_iter == 0:
                start_iter = hp.SAVEITER
                end_iter = hp.SAVEITER*60
            else:
                start_iter = specific_iter
                end_iter = specific_iter
            print('Testing...')

        elif mode == 'valid':
            reuse = True
            start_iter = specific_iter
            end_iter = specific_iter
        #Modify for adding ppg 
        if self.ppg_path != None:
            input_dim = hp.f_bin + hp.ppg_dim 
        else:
            input_dim = hp.f_bin
        inputs   = tf.placeholder("float", [1, 1, input_dim, None], name='input_noisy')
        enhanced = self.se(inputs, reuse=reuse, is_training=False)

        _pathname = os.path.join(test_path, "enhanced")
        check_dir(_pathname)       
        nlist = [x[:-1] for x in open(test_list,'r').readlines()]
        for latest_step in range(start_iter, end_iter+hp.SAVEITER, hp.SAVEITER):
            ### Load model    
            if mode == 'test':
                saver = tf.train.Saver(self.se.vars)
                saver.restore(self.sess, test_path + "model-" + str(latest_step))
                print(latest_step)

            ### Enhanced waves in the nlist
            enh_list = []
            mse_list = []
            for name in nlist:
                if name == '':
                    continue
                _pathname = os.path.join(test_path,"enhanced",str(latest_step))
                check_dir(_pathname)
                ### Read file                   
                spec, phase, x = make_spectrum(name, is_slice=False, feature_type=hp.feature_type, mode=hp.nfeature_mode)                    
                #Add ppg into spec
                if self.ppg_path != None:
                    ppg_name = name.split('/')[-1].split('.')[0] + '.txt' 
                    if mode == 'test':
                        ppg_name = name.split('/')[-1].split('.')[0] + "_" + name.split('/')[-3] + "_" + name.split('/')[-2] + '.txt' 
                    # ppg_spec = make_ppg_spectrum(ppg_name,self.ppg_data,is_slice=False)
                    ppg_spec = make_ppg_btn_spectrum(ppg_name,self.ppg_path,is_slice=False,feature_type=hp.feature_type, mode=hp.nfeature_mode)
                    spec = np.append(spec,ppg_spec,axis = 0)

                spec_length = spec.shape[1]

                ''' run sliced spectrogram '''
                ## Pad spectrogram
                # temp = np.zeros((spec.shape[0], ((spec_length-FRAMELENGTH)//OVERLAP+1)*OVERLAP+FRAMELENGTH))
                # temp[:,:spec_length] = spec                  
                # ### Slice spectrogram into segments
                # slices = []
                # for i in range(0, temp.shape[1]-FRAMELENGTH+1, OVERLAP):
                #     slices.append(temp[:,i:i+FRAMELENGTH])
                # slices = np.array(slices).reshape((-1, 1, 257, self.frame_size))
                # ### Run graph
                # spec_temp = np.zeros(temp.shape)
                # output = self.sess.run(enhanced, {inputs:slices})

                # for i,out_frame in enumerate(output):
                #     spec_temp[:,i*OVERLAP:i*OVERLAP+FRAMELENGTH] += out_frame[0,:,:]
                # spec_temp = spec_temp[:,:spec_length] 


                ''' run whole spectrogram '''
                spec = np.reshape(spec, [1, 1, -1, spec_length])                    
                output = self.sess.run(enhanced, {inputs:spec})
                spec_temp = output[0, 0, :, :]

                recons_y = recons_spec_phase(spec_temp, phase, feature_type=hp.feature_type) 
                y_out = librosa.util.fix_length(recons_y, x.shape[0])

                temp_name = name.split('/')
                _pathname = os.path.join(_pathname,temp_name[-4])
                check_dir(_pathname)
                _pathname = os.path.join(_pathname,temp_name[-3])
                check_dir(_pathname)
                _pathname = os.path.join(_pathname,temp_name[-2])
                check_dir(_pathname)
                _pathname = os.path.join(_pathname,temp_name[-1])
                wav.write(_pathname, 16000, np.int16(y_out*32767))
                enh_list.append(_pathname)

            # pesq = read_batch_pesq(clean_root, enh_list) 
            # print("Avg PESQ: {}, MSE: {}".format(np.mean(pesq), np.mean(mse_list)))

        return 
