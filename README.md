# BPCSE
Using BPC-ASR to improve SE

=================================
1.ASR TRAINING
step1: prepare timit data
1-a: copy a timit & prepare broad phone .PHN
cp -r TIMIT TIMIT_Broad_Place
sudo python pre/timit/phn2broad.py

1-b change noisy .wav to sphere 
sudo bash pre/timit/wav2sph.sh

1-c: prepare noisy .WAV
sudo bash pre/timit/clean2noisy.sh

step2: train broad noisy timit
cd ASR/kaldi
sudo bash train.sh

=================================
2.ASR INFERENCE
step1: prepare TIMIT data
1-a: change noisy .wav to sphere 
sh pre/timit/wav2sph.sh

1-b: prepare kaldi info
python pre/timit/prepare_kaldi.py

step2: inference broad noisy TIMIT
cd ASR/kaldi
sudo bash inference.sh

