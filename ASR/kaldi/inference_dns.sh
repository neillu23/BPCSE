ASR_PATH="/home/neillu/Desktop/ASR-DNS/ASR-DNS/"
KALDI_PATH="/home/neillu/Desktop/Workspace/Kaldi_5Broad_frame/kaldi/egs/timit/s5"

#prepare codes
cp inference/inf_run.sh inference/inf_run_dnn.sh inference/inf_ppg.sh $KALDI_PATH
cd $KALDI_PATH

#Prepare extract datas
rm -r data/extract
cp -r $ASR_PATH/ASR/pre/dns/extract data/extract/
./utils/fix_data_dir.sh data/extract/ 

#run.sh
sudo bash inf_run.sh extract

#run_dnn.sh
sudo bash inf_run_dnn.sh extract

#extract feature
sudo bash inf_ppg.sh extract
