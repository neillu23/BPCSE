import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io.wavfile as wav
import librosa
import random, os
from sklearn import preprocessing
import glob
from PIL import Image
from tqdm import tqdm
import tensorflow as tf

from hyperparameters import Hyperparams as hp


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def slice_pad(spec, OVERLAP, seg_size=64, pad_value=0):
    ### Pad spectrogram
    F, T = spec.shape
    temp = np.ones([F, ((T-seg_size)//OVERLAP+1)*OVERLAP+seg_size], dtype=spec.dtype) * pad_value
    temp[:,:T] = spec  
    ### Slice spectrogram into segments
    slices = []
    for i in range(0, temp.shape[1]-seg_size+1, OVERLAP):
        slices.append(temp[:,i:i+seg_size])
    slices = np.array(slices).reshape((-1, 1, F, seg_size))
    return slices

def make_spectrum(filename=None, y=None, is_slice=False, feature_type='logmag', mode=None, FRAMELENGTH=None, SHIFT=None, _max=None, _min=None):
    '''
    Return:
    Sxx   = [F, T] (is_slice==False) or [T//FRAMELENGTH, F, FRAMELENGTH] (is_slice==True)
    phase = [F, T] (is_slice==False) or [T//FRAMELENGTH, F, FRAMELENGTH] (is_slice==True)
    y     = y

    '''
    if y is not None:
        y = y
    else:
        y, sr = librosa.load(filename, sr=16000)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype == 'int16':
            y = np.float32(y/32767.)
        elif y.dtype !='float32':
            y = np.float32(y)

    ### Normalize waveform
    # y = y / np.max(abs(y)) / 2.

    D = librosa.stft(y,center=False, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.n_fft, window=scipy.signal.hamming)
    utt_len = D.shape[-1]
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    ### Feature type
    if feature_type == 'logmag':
        Sxx = np.log1p(D)
    elif feature_type == 'lps':
        Sxx = np.log10(D**2)
    else:
        Sxx = D

    if mode == 'mean_std':
        mean = np.mean(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))
        std = np.std(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))+1e-12
        Sxx = (Sxx-mean)/std  
    elif mode == 'minmax':
        Sxx = 2 * (Sxx - _min)/(_max - _min) - 1


    # print("noisy_spec.shape before slice_pad:",Sxx.shape)
    if is_slice:
        Sxx = slice_pad(Sxx, SHIFT, seg_size=FRAMELENGTH, pad_value=0)
    # print("noisy_spec.shape after slice_pad:",Sxx.shape)

    return Sxx, phase, y

# def load_ppg_spectrum(ppg_path):
#     #TODO: read name ppg_data in ppg_path files
#     ppg_data = {}
#     for root, dirs, files in os.walk(ppg_path):
#         for ppg_file in files:
#             with open(os.path.join(root,ppg_file),'r') as fp:
#                 lines = fp.readlines()
#                 name = ""
#                 start_id = 0
#                 end_id =0
#                 for idx, line in enumerate(lines):
#                     if '[' in line:
#                         name = line.split()[0]
#                         start_id = idx + 1
#                     if start_id != None and "]" in line:
#                         end_id = idx
#                         ppg_spec = np.array([np.array([float(ele) for ele in line.split(" ")[2:-1]]) for line in lines[start_id:end_id+1]])
#                         ppg_spec = ppg_spec.astype(np.float32)
#                         ppg_data[name] = ppg_spec.T

#         return ppg_data

# def make_ppg_spectrum(name,ppg_data, is_slice=False, FRAMELENGTH=None, SHIFT=None):
#     ppg_spec = ppg_data[name.upper()]
#     if is_slice:
#         # print("ppg_spec.shape before slice_pad:",ppg_spec.shape)
#         ppg_spec = slice_pad(ppg_spec, SHIFT, seg_size=FRAMELENGTH, pad_value=0)
#         # print("ppg_spec.shape before slice_pad:",ppg_spec.shape)

#     return np.array(ppg_spec)

def make_ppg_btn_spectrum(name,ppg_path,feature_type='logmag', mode=None, is_slice=False, FRAMELENGTH=None, SHIFT=None, _max=None, _min=None):
    ppg_spec = np.loadtxt(os.path.join(ppg_path,name), delimiter = " ")
    ppg_spec = ppg_spec.astype(np.float32)
    ppg_spec = ppg_spec.T

    # ### Feature type
    # if feature_type == 'logmag':
    #     Sxx = np.log1p(D)
    # elif feature_type == 'lps':
    #     Sxx = np.log10(D**2)
    # else:
    #     Sxx = D

    # Feature type: sigmoid
    ppg_spec = sigmoid(ppg_spec)


    if mode == 'mean_std':
        mean = np.mean(ppg_spec, axis=1).reshape((hp.ppg_dim, 1))
        std = np.std(ppg_spec, axis=1).reshape((hp.ppg_dim, 1))+1e-12
        ppg_spec = (ppg_spec-mean)/std  
    elif mode == 'minmax':
        ppg_spec = 2 * (ppg_spec - _min)/(_max - _min) - 1

    if is_slice:
        # print("ppg_spec.shape before slice_pad:",ppg_spec.shape)
        ppg_spec = slice_pad(ppg_spec, SHIFT, seg_size=FRAMELENGTH, pad_value=0)
        # print("ppg_spec.shape before slice_pad:",ppg_spec.shape)

    return np.array(ppg_spec)

    # #read fileid ppg_spec in ppg_path files
    # 
    # for root, dirs, files in os.walk(ppg_path):
    #     for ppg_file in files:
    #         with open(os.path.join(root,ppg_file),'r') as fp:
    #             start_id = None
    #             lines = fp.readlines()
    #             for idx, line in enumerate(lines):
    #                 if name.upper() in line:
    #                     start_id = idx + 1
    #                 if start_id != None and "]" in line:
    #                     end_id = idx
    #                     break
    #             ppg_spec = np.array([np.array([float(ele) for ele in line.split(" ")[2:-1]]) for line in lines[start_id:end_id+1]])
    #             ppg_spec = ppg_spec.T
    #             break

def recons_spec_phase(Sxx_r, phase, feature_type='logmag'):
    if feature_type == 'logmag':
        Sxx_r = np.expm1(Sxx_r)
        if np.min(Sxx_r) < 0:
            print("Expm1 < 0 !!")
        Sxx_r = np.clip(Sxx_r, a_min=0., a_max=None)
    elif feature_type == 'lps':
        Sxx_r = np.sqrt(10**Sxx_r)

    R = np.multiply(Sxx_r , phase)
    result = librosa.istft(R,
                     center=False,
                     hop_length=hp.hop_length,
                     win_length=hp.n_fft,
                     window=scipy.signal.hamming)
    return result

# From https://github.com/candlewill/Griffin_lim/blob/master/utils/audio.py
def griffinlim(spectrogram, n_iter = 100, n_fft = 512, hop_length = 256):
    # spectrogram = np.sqrt(10**spectrogram)
    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    for i in range(n_iter):
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length = hop_length, window = scipy.signal.hamming)
        rebuilt = librosa.stft(inverse, n_fft = n_fft, hop_length = hop_length, window = scipy.signal.hamming)
        angles = np.exp(1j * np.angle(rebuilt))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length = hop_length, window = scipy.signal.hamming)

    return inverse

class dataPreprocessor(object):
    def __init__(self, 
                record_path, record_name,
                noisy_list=None, 
                clean_path=None,
                ppg_path=None,
                use_waveform=True, 
                frame_size=64, 
                shift=None):
        self.noisy_list = noisy_list
        self.ppg_path = ppg_path
        self.ppg_data = {}
        self.clean_path = clean_path
        self.use_waveform = use_waveform
        self.record_path = record_path
        self.record_name = record_name

        self.FRAMELENGTH = frame_size
        self.SHIFT = frame_size if shift == None else shift

    # def get_speaker_dict(self, all_cln_files_name):
    #     spker_dict = dict()
    #     idx = 0
    #     for file_name in all_cln_files_name:
    #         name = file_name.split('/')[-1].split('_')[0] ### /path/to/file/{spker}_{sentence}.wav for TIMIT data
    #         if name not in spker_dict:
    #             spker_dict[name] = idx
    #             idx += 1
    #     return spker_dict

    def write_tfrecord(self):
        if tf.gfile.Exists(self.record_path):
            print('Folder already exists: {}\n'.format(self.record_path))
        else:
            tf.gfile.MkDir(self.record_path)

        n_files = np.array([x[:-1] for x in open(self.noisy_list).readlines()])
        # ### Shuffle it first
        # shuffle_id = np.arange(len(n_files))
        # random.shuffle(shuffle_id)
        # n_files = n_files[shuffle_id]
        # n_files = n_files[:10]

        out_file = tf.python_io.TFRecordWriter(self.record_path+self.record_name+'.tfrecord')
        if self.use_waveform:
            print("Not compatible on waveform for now!!")
        else:
            print("{} spectrogram!".format(hp.feature_type))
            cnt1 = 0
            # if self.ppg_path != None:
            #     self.ppg_data = load_ppg_spectrum(self.ppg_path)
            for i,n_ in enumerate(tqdm(n_files)):
                ### use noisy filename to find clean file
                name = n_.split('/')[-1].split('_')[0] + '_' + n_.split('/')[-1].split('_')[1] + ".wav"
                c_ = os.path.join(self.clean_path, name)
                # print(n_, c_)
                noisy_spec,_,_ = make_spectrum(n_,
                                                is_slice=True,
                                                feature_type=hp.feature_type, 
                                                mode=hp.nfeature_mode, 
                                                FRAMELENGTH=self.FRAMELENGTH, 
                                                SHIFT=self.SHIFT)
                # add for adding ppg
                # print("noisy_spec.shape:",noisy_spec.shape)
                if self.ppg_path != None:
                    ppg_spec = make_ppg_btn_spectrum(
                                                os.path.join(n_.split('/')[-1].split('.')[0] + ".txt"),
                                                self.ppg_path,
                                                is_slice=True,
                                                feature_type=hp.feature_type, 
                                                mode=hp.nfeature_mode, 
                                                FRAMELENGTH=self.FRAMELENGTH, 
                                                SHIFT=self.SHIFT)
                    # ppg_spec = make_ppg_spectrum(name.split('.')[0],
                    #                             self.ppg_data,
                    #                             is_slice=True,
                    #                             FRAMELENGTH=self.FRAMELENGTH, 
                    #                             SHIFT=self.SHIFT)
                    # print("noisy_spec.shape:",noisy_spec.shape)
                    # print("ppg_spec.shape:",ppg_spec.shape)
                    noisy_spec = np.append(noisy_spec,ppg_spec,axis = 2)
                    # print("new noisy_spec.shape:",noisy_spec.shape)
                clean_spec,_,_ = make_spectrum(c_,
                                                is_slice=True, 
                                                feature_type=hp.feature_type, 
                                                mode=None, 
                                                FRAMELENGTH=self.FRAMELENGTH, 
                                                SHIFT=self.SHIFT)
                for n_spec,c_spec in zip(
                                        noisy_spec, 
                                        clean_spec, 
                                        ):
                    cnt1 += 1
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'n_spec': _bytes_feature(n_spec.tostring()),
                        'c_spec': _bytes_feature(c_spec.tostring()),  
                        }))

                    out_file.write(example.SerializeToString())

            out_file.close()  

            print("num_samples = %d"%cnt1)

    def read_and_decode(self,batch_size=16, num_threads=16):
        filename_queue = tf.train.string_input_producer([self.record_path+self.record_name+'.tfrecord'])        
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
                serialized_example,
                features={
                    'n_spec': tf.FixedLenFeature([], tf.string),
                    'c_spec': tf.FixedLenFeature([], tf.string),
                })

        if self.use_waveform:
            print("Not compatible on waveform for now!!")
        else:
            wave = tf.decode_raw(features['c_spec'], tf.float32, name='data_clean')
            # print("ori wave shape: ", wave)  
            wave = tf.reshape(wave, [1, hp.f_bin, self.FRAMELENGTH])
            # print("wave shape: ", wave)    
            noisy = tf.decode_raw(features['n_spec'], tf.float32, name='data_noisy')
            # print("ori noisy shape: ", noisy)
            #Modify for adding ppg 
            if self.ppg_path != None:
                input_dim = hp.f_bin + hp.ppg_dim 
            else:
                input_dim = hp.f_bin
            noisy = tf.reshape(noisy, [1, input_dim, self.FRAMELENGTH])  
            # print("noisy shape: ", noisy)          
            return tf.train.shuffle_batch(
                                        [wave, noisy],
                                        batch_size=batch_size,
                                        num_threads=num_threads,
                                        capacity=1000 + 10 * batch_size,
                                        min_after_dequeue=1000,
                                        name='wav_and_label')
 

def save_image(img, path):
    _max = np.max(img)
    _min = np.min(img)
    # print(_max , _min)
    img = (img - _min)/(_max - _min) * 2 - 1 
    I8 = ((img+1.) * 128).astype(np.uint8)
    img = Image.fromarray(I8[::-1])
    img.save(path+".png")

def merge_save_images(images, size, path):
    images = np.array(images)
    h, w = images.shape[1], images.shape[2]
    expand_images = np.ones((images.shape[0], h, w+5))
    expand_images[:,:,:w] = images
    w += 5
    img = np.zeros((h * size[0], (w) * size[1]))
    for idx, image in enumerate(expand_images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w] = image[:,:]

    save_image(img, path)
