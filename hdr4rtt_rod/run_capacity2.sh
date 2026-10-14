#!/bin/bash
# Adds the like-for-like tone-map cell for RetinaNet. The capacity runs give
# TMM x {RetinaNet, FasterRCNN}. The detector table's fine-tuned FasterRCNN
# (43.0) is NOT a matching cell: it used gamma at p99, not Reinhard, and was
# trained by train_torchvision.py for 12 epochs at batch 2. This run is
# Reinhard + RetinaNet through train_frontend.py with the capacity runs'
# settings, without which "detector" and "front end" stay tangled for the
# RetinaNet row. A matching Reinhard + FasterRCNN cell has not been run.
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
