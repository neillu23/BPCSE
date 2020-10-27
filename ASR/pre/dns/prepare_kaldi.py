import os 
from os import path

sph2pipe = "/home/neillu/Desktop/Kaldi_5Broad/kaldi/tools/sph2pipe_v2.5/sph2pipe"
wav_path = "/home/neillu/Desktop/Workspace/DNS-Challenge-227/DNS-Challenge/training/noisy_sph"
cor_path = "/home/neillu/Desktop/ASR-DNS/ASR-DNS/data/Correct_List"
des_path = "extract/"

def parse_wav(wav_path,cor_path):
    uttlist = {}
    spklist = {}
    spkgen = {}
    # genlist = {}

    #Get speaker information
    for root, dirs, files in os.walk(wav_path):
        # print root, dirs, files
        for wav_file in files:
            utt_name = wav_file.split(".")[0]
            strs = utt_name.split("_")
            spk = strs[strs.index("reader")+1]
            utt_name = spk + utt_name
            uttlist[utt_name] = (root,wav_file,spk)
            if spk in spklist:
                spklist[spk].append(utt_name)
                # assert(spkgen[spk] == genlist[utt_name])
            else:
                spklist[spk] = [utt_name]
                # spkgen[spk] = genlist[utt_name]

    #Initial spkgen (TODO: Some of speaker not in corr list ex:00087)
    for spk in spklist:
        spkgen[spk] = "m"

    #Get Gender information
    for root, dirs, files in os.walk(cor_path):
        for cor_file in files:
            if "clean" in cor_file:
                continue
            gend = cor_file.split("_")[1][0]
            with open(path.join(root,cor_file),'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    strs = line.split("/")[-1].split("_")
                    spk = strs[strs.index("reader")+1]
                    spkgen[spk] = gend

    return uttlist, spklist, spkgen

def gen_wav_scp(uttlist,sph2pipe,wav_file):
    with open(wav_file,'w') as fp:
        for utt in uttlist:
            fp.write(utt + " " + sph2pipe + " -f wav " + path.join(uttlist[utt][0],uttlist[utt][1]) + " |\n")


def gen_utt2spk(uttlist,utt_file):
    with open(utt_file,'w') as fp:
        for utt in uttlist:
            fp.write(utt + " " + uttlist[utt][2] + "\n")
        # for spk in spklist:
        #     for utt in spklist[spk]:
        #         fp.write(utt + " " + spk + "\n")
    
def gen_spk2utt(spklist,spk_file):
    with open(spk_file,'w') as fp:
        for spk in spklist:
            fp.write(spk)
            for utt in spklist[spk]:
                fp.write(" " + utt)
            fp.write("\n")
    


def gen_spk2gender(spkgen, gen_file):
    with open(gen_file,'w') as fp:
        for spk in spkgen:
            fp.write(spk + " " + spkgen[spk] + "\n")
    


uttlist, spklist, spkgen = parse_wav(wav_path,cor_path)
 
gen_wav_scp(uttlist,sph2pipe,path.join(des_path,"wav.scp"))
gen_utt2spk(uttlist,path.join(des_path,"utt2spk"))
gen_spk2utt(spklist,path.join(des_path,"spk2utt"))
gen_spk2gender(spkgen,path.join(des_path,"spk2gender"))
