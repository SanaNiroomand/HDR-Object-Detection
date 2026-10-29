#!/bin/bash
# Two questions on the 2-class Faster R-CNN split, settings exactly as in
# run_capacity3.sh (10 epochs, batch 4), where global Reinhard scored 44.9.
#
# 1. Local exposure. Plain local Reinhard (run_local_reinhard.sh) scored 44.3:
#    it only changes bright pixels, so the dark, flat objects the failure
#    analysis points at come out unchanged. reinhard_localexp lifts dark regions
#    and lowers bright ones before the same global curve.
# 2. Noise. Every figure so far is one run. The trainer sets no seed, so
#    repeating global Reinhard twice under new tags gives three independent runs
#    and a first estimate of how far mAP moves by chance alone.
PY="/c/Users/OGAM/miniconda3/envs/hs-ml/python.exe"
ANN="D:/Data/HDR/hdr4rtt_rod/annotations"
cd "D:/Codes/HDR/Sana/hdr4rtt_rod"

run_cell() {   # arm, tag, output json
  echo "##### TRAIN $1 ($2) #####"
  "$PY" train_frontend.py --arm "$1" --arch fasterrcnn --epochs 10 --batch 4 --workers 4 \
    --tag "$2" --train_ann "$ANN/hdr4rtt_rod_dedup_train.json" \
    --val_ann "$ANN/hdr4rtt_rod_dedup_test.json" \
    || { echo "##### TRAIN FAILED $1 $2 #####"; return 1; }
  echo "##### EVAL $1 ($2) #####"
  "$PY" eval_frontend.py --arm "$1" --arch fasterrcnn --batch 4 --tag "$2" \
    --ann "$ANN/hdr4rtt_rod_dedup_test.json" --out_json "$3" \
    || { echo "##### EVAL FAILED $1 $2 #####"; return 1; }
}

echo "##### BUILD reinhard_localexp images #####"
"$PY" make_frontends.py --arms reinhard_localexp --workers 10 \
  && run_cell reinhard_localexp rod2cls results/capacity/fasterrcnn_reinhard_localexp_rod2cls.json \
  || echo "##### LOCAL EXPOSURE SKIPPED OR FAILED #####"
run_cell reinhard rod2cls_rep2 results/capacity/fasterrcnn_reinhard_rod2cls_rep2.json
run_cell reinhard rod2cls_rep3 results/capacity/fasterrcnn_reinhard_rod2cls_rep3.json
echo "##### ALL COMPLETE #####"
