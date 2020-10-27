
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)



extract_folder=$1
TYPE=$2
# Config:

if [ $TYPE == "mono" ]; then
  gmmdir=exp/mono
  data_fmllr=data
  # data_fmllr=data-fmllr-mono2
else
  gmmdir=exp/tri3
  data_fmllr=data-fmllr-tri3
fi

stage=0 # resume training with --stage=N
# End of config.
. utils/parse_options.sh || exit 1;
#


if [ $TYPE == "tri" ]; then
  if [ $stage -le 0 ]; then
    # Store fMLLR features, so we can train on them easily,
    # extract
    dir=$data_fmllr/$extract_folder
    steps/nnet/make_fmllr_feats.sh --nj 20 --cmd "$train_cmd" \
      --transform-dir $gmmdir/decode_$extract_folder \
      $dir data/$extract_folder $gmmdir $dir/log $dir/data || exit 1
  fi
fi


dir=exp/dnn4_pretrain-dbn_dnn_smbr
srcdir=exp/dnn4_pretrain-dbn_dnn
acwt=0.2

if [ $stage -le 4 ]; then
  for ITER in 6; do  
    copy-feats scp:$data_fmllr/$extract_folder/feats.scp ark:feature.ark
    nnet-forward --no-softmax=true --prior-scale=1.0 --feature-transform=exp/dnn4_pretrain-dbn_dnn/final.feature_transform --class-frame-counts=exp/dnn4_pretrain-dbn_dnn/ali_train_pdf.counts --use-gpu=no $dir/${ITER}.nnet ark:feature.ark ark:ppg.ark
    mkdir $extract_folder
    copy-feats ark:ppg.ark ark,scp,t:$extract_folder/ppg.ark,$extract_folder/ppg.scp
  done 
fi


echo Success
exit 0


