ASR_PATH="/home/neillu/Desktop/ASR-DNS/ASR-DNS/"
KALDI_PATH="/home/neillu/Desktop/Workspace/Kaldi_Broad_Place_frame_mono/kaldi/egs/timit/s5"
TYPE="mono" #"tri" or "mono"

#prepare codes
cp training/run.sh training/cmd.sh training/dns_mfcc_config.txt $KALDI_PATH
cp training/run_dnn.sh $KALDI_PATH/local/nnet/

cd $KALDI_PATH


#run.sh
sudo bash run.sh $TYPE

#run_dnn.sh
sudo bash local/nnet/run_dnn.sh $TYPE
