#!/bin/bash
# Local versus global Reinhard, behind the same Faster R-CNN, like for like.
#
# Global Reinhard divides each pixel by itself, L / (1 + L). The local operator
# from the same 2002 paper divides by a blurred neighbourhood, L / (1 + V1), with
# the blur size chosen per pixel. The failure analysis found misses track
# contrast inside the object's own box, so a local operator is the first thing to
# test. Settings match run_capacity3.sh exactly (10 epochs, batch 4, 2-class
# deduplicated split), where global Reinhard scored 44.9.
PY="/c/Users/OGAM/miniconda3/envs/hs-ml/python.exe"
ANN="D:/Data/HDR/hdr4rtt_rod/annotations"
cd "D:/Codes/HDR/Sana/hdr4rtt_rod"
echo "##### BUILD reinhard_local images #####"
"$PY" make_frontends.py --arms reinhard_local --workers 10 \
  || { echo "##### BUILD FAILED #####"; exit 1; }
echo "##### TRAIN reinhard_local + fasterrcnn (2-class ROD split) #####"
"$PY" train_frontend.py --arm reinhard_local --arch fasterrcnn --epochs 10 --batch 4 --workers 4 \
  --tag rod2cls --train_ann "$ANN/hdr4rtt_rod_dedup_train.json" \
  --val_ann "$ANN/hdr4rtt_rod_dedup_test.json" \
  || { echo "##### TRAIN FAILED #####"; exit 1; }
echo "##### EVAL reinhard_local + fasterrcnn #####"
"$PY" eval_frontend.py --arm reinhard_local --arch fasterrcnn --batch 4 --tag rod2cls \
  --ann "$ANN/hdr4rtt_rod_dedup_test.json" \
  --out_json results/capacity/fasterrcnn_reinhard_local_rod2cls.json \
  || { echo "##### EVAL FAILED #####"; exit 1; }
echo "##### LOCAL REINHARD COMPLETE #####"
