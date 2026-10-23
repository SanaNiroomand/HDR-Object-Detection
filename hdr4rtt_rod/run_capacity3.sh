#!/bin/bash
# The missing like-for-like cell of the capacity table: Reinhard + FasterRCNN
# through train_frontend.py with exactly the capacity runs' settings (10 epochs,
# batch 4, 2-class deduplicated split).
#
# Until this existed, the FasterRCNN row compared RAOD's module (42.6) against
# the detector table's tone-mapped FasterRCNN (43.0), which used gamma at p99 and
# was trained by train_torchvision.py for 12 epochs at batch 2. Two things
# differed besides the front end, so the 0.4 gap was not a controlled measurement.
PY="/c/Users/OGAM/miniconda3/envs/hs-ml/python.exe"
ANN="D:/Data/HDR/hdr4rtt_rod/annotations"
cd "D:/Codes/HDR/Sana/hdr4rtt_rod"
echo "##### TRAIN reinhard + fasterrcnn (2-class ROD split) #####"
"$PY" train_frontend.py --arm reinhard --arch fasterrcnn --epochs 10 --batch 4 --workers 4 \
  --tag rod2cls --train_ann "$ANN/hdr4rtt_rod_dedup_train.json" \
  --val_ann "$ANN/hdr4rtt_rod_dedup_test.json" \
  || { echo "##### TRAIN FAILED reinhard #####"; exit 1; }
echo "##### EVAL reinhard + fasterrcnn #####"
"$PY" eval_frontend.py --arm reinhard --arch fasterrcnn --batch 4 --tag rod2cls \
  --ann "$ANN/hdr4rtt_rod_dedup_test.json" \
  --out_json results/capacity/fasterrcnn_reinhard_rod2cls.json \
  || { echo "##### EVAL FAILED reinhard #####"; exit 1; }
echo "##### CELL COMPLETE #####"
