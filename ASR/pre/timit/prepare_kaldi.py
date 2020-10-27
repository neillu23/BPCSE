import os 
from os import path

sph2pipe = "/home/neillu/Desktop/Kaldi_5Broad/kaldi/tools/sph2pipe_v2.5/sph2pipe"
wav_path = "/home/neillu/Downloads/TIMIT_5Broad"
des_path = "extract_timit/"

def parse_wav(wav_path):
    uttlist = {}
    spklist = {}
    spkgen = {}
    # genlist = {}

    #Get speaker information
    for root, dirs, files in os.walk(wav_path):
        # print root, dirs, files
        for wav_file in files:
            if ".WAV" not in wav_file :
                continue
            spk = root.split("/")[-1]
            utt_name = spk + "_" + wav_file.split(".")[0]
            uttlist[utt_name] = (root,wav_file,spk)
            if spk in spklist:
                spklist[spk].append(utt_name)
                # assert(spkgen[spk] == genlist[utt_name])
            else:
                spklist[spk] = [utt_name]
                # spkgen[spk] = genlist[utt_name]

    #Get speake gender
    for spk in spklist:
        spkgen[spk] = spk[0].lower()

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
    


uttlist, spklist, spkgen = parse_wav(wav_path)
 
gen_wav_scp(uttlist,sph2pipe,path.join(des_path,"wav.scp"))
gen_utt2spk(uttlist,path.join(des_path,"utt2spk"))
gen_spk2utt(spklist,path.join(des_path,"spk2utt"))
gen_spk2gender(spkgen,path.join(des_path,"spk2gender"))
