ASR_PATH="/home/neillu/Desktop/ASR-DNS/ASR-DNS/"
KALDI_PATH="/home/neillu/Desktop/Workspace/Kaldi_Broad_Place_frame_mono/kaldi/egs/timit/s5"
TYPE="mono" #TYPE = "tri" or "mono"

#prepare codes
cp inference/inf_run.sh inference/inf_run_dnn.sh training/dns_mfcc_config.txt $KALDI_PATH
cd $KALDI_PATH

#Prepare extract datas
rm -r data/extract_timit
cp -r $ASR_PATH/ASR/pre/timit/extract_timit data/extract_timit/
./utils/fix_data_dir.sh data/extract_timit/ 

#run.sh
sudo bash inf_run.sh extract_timit $TYPE

#run_dnn.sh & extract feature
sudo bash inf_run_dnn.sh extract_timit $TYPE

