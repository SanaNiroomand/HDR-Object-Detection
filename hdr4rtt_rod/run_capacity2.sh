#!/bin/bash
# Completes the 2x2. The capacity runs give TMM x {RetinaNet, FasterRCNN}; the
# detector table already has Reinhard + FasterRCNN fine-tuned at 43.0. The one
# missing cell is Reinhard + RetinaNet on the same 2-class split, without which
# "detector" and "front end" stay tangled for the RetinaNet row.
PY="/c/Users/OGAM/miniconda3/envs/hs-ml/python.exe"
ANN="D:/Data/HDR/hdr4rtt_rod/annotations"
cd "D:/Codes/HDR/Sana/hdr4rtt_rod"
until grep -q "CAPACITY RUNS COMPLETE" capacity.log; do sleep 30; done
echo "##### TRAIN reinhard + retinanet (2-class ROD split) #####"
"$PY" train_frontend.py --arm reinhard --arch retinanet --epochs 10 --batch 4 --workers 4 \
  --tag rod2cls --train_ann "$ANN/hdr4rtt_rod_dedup_train.json" \
  --val_ann "$ANN/hdr4rtt_rod_dedup_test.json" \
  || { echo "##### TRAIN FAILED reinhard #####"; exit 1; }
echo "##### EVAL reinhard + retinanet #####"
"$PY" eval_frontend.py --arm reinhard --arch retinanet --batch 4 --tag rod2cls \
  --ann "$ANN/hdr4rtt_rod_dedup_test.json" || echo "##### EVAL FAILED reinhard #####"
echo "##### SQUARE COMPLETE #####"
