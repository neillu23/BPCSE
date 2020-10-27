import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
from pypesq import pesq
from pystoi import stoi
import sys

os.environ["CUDA_VISIBLE_DEVICES"]="1"

noisy_list="/mnt/Data/user_vol_2/user_neillu/BPCSE/ts_timit"
clean_path="/mnt/Corpus/TIMIT_wav_noisy/testing/test_clean/"

out_name = sys.argv[1]

result_path = "/mnt/Data/user_vol_2/user_neillu/score/" + out_name
log_path = "/mnt/Data/user_vol_2/user_neillu/score/all_result/" 

if not os.path.exists(result_path):
    os.mkdir(result_path)

if not os.path.exists(log_path):
    os.mkdir(log_path)

n_files = np.array([x[:-1] for x in open(noisy_list).readlines()])
stoi_score_list = {"n10dB":[],"n5dB":[],"0dB":[],"5dB":[],"10dB":[],"15dB":[],"20dB":[]}
pesq_score_list = {"n10dB":[],"n5dB":[],"0dB":[],"5dB":[],"10dB":[],"15dB":[],"20dB":[]}
for i,n_ in enumerate(tqdm(n_files)):
    clean_name = n_.split('/')[-1].lower()
    snr = n_.split('/')[-2]
    c_ = os.path.join(clean_path, clean_name)
    e_ = n_ #os.path.join(enhance_path, noisy_path)
    clean, fs = sf.read(c_)
    denoised, fs = sf.read(e_)
    # Clean and den should have the same length, and be 1D
    stoi_score = stoi(clean, denoised, fs, extended=False)
    pesq_score = pesq(clean, denoised, fs)
    stoi_score_list[snr].append(stoi_score)
    pesq_score_list[snr].append(pesq_score)
    # print("stoi:",stoi_score)
    # print("pesq:",pesq_score)
fp  = open(os.path.join(log_path,out_name+".csv"),"w")
fp.write("SNR, pesq_avg, stoi_avg"+"\n")
for key in stoi_score_list:
    np.savetxt(os.path.join(result_path,"stoi_"+key+".txt"), np.array(stoi_score_list[key]))
    np.savetxt(os.path.join(result_path,"pesq_"+key+".txt"), np.array(pesq_score_list[key]))
    print("SNR"+key,np.average(pesq_score_list[key]),np.average(stoi_score_list[key]))
    fp.write("SNR"+key+", "+str(np.average(pesq_score_list[key]))+", "+str(np.average(stoi_score_list[key]))+"\n")
