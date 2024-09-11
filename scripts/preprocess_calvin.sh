rank="$1"
bsz="$2"
CUDA_VISIBLE_DEVICES=0 python3 src/preprocess_calvin_penultimate.py -r "$rank" -b "$bsz"