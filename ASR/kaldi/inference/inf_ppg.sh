. ./cmd.sh
. ./path.sh
copy-feats scp:data-fmllr-tri3/extract/split5/1/feats.scp ark:feature.ark
nnet-forward --no-softmax=true --prior-scale=1.0 --feature-transform=exp/dnn4_pretrain-dbn_dnn/final.feature_transform --class-frame-counts=exp/dnn4_pretrain-dbn_dnn/ali_train_pdf.counts --use-gpu=no exp/dnn4_pretrain-dbn_dnn/final.nnet ark:feature.ark ark:ppg.ark
copy-feats ark:ppg.ark ark,scp,t:ppg_2.ark,ppg_2.scp
copy-feats-to-htk --output-ext=mfc scp:ppg_2.scp
