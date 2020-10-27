
. ./cmd.sh
[ -f path.sh ] && . ./path.sh
set -e

# Acoustic model parameters
numLeavesTri1=2500
numGaussTri1=15000
numLeavesMLLT=2500
numGaussMLLT=15000
numLeavesSAT=2500
numGaussSAT=15000
numGaussUBM=400
numLeavesSGMM=7000
numGaussSGMM=9000

feats_nj=10
train_nj=30
decode_nj=20

extract_folder=$1
TYPE=$2

echo ============================================================================
echo "         MFCC Feature Extration & CMVN for Training and Test set          "
echo ============================================================================

# Now make MFCC features.
mfccdir=mfcc

for x in $extract_folder; do
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $feats_nj --mfcc-config dns_mfcc_config.txt data/$x exp/make_mfcc/$x $mfccdir
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
done

if [ $TYPE == "mono" ]; then
  echo ============================================================================
  echo "                     MonoPhone Training & Decoding                        "
  echo ============================================================================

  steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
  exp/mono/graph data/$extract_folder exp/mono/decode_$extract_folder

  # steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
  # exp/mono/graph data/$extract_folder exp/mono2/decode_$extract_folder

else
  echo ============================================================================
  echo "              tri3 : LDA + MLLT + SAT Training & Decoding                 "
  echo ============================================================================

  steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" --skip-scoring true\
  exp/tri3/graph data/$extract_folder exp/tri3/decode_$extract_folder 
fi
