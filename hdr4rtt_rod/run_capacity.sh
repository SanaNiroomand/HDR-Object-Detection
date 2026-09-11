#!/bin/bash
# Does RAOD's learned tone map fail because the MODULE is weak, or because the
# detector behind it is 1M parameters?
#
# The detector-comparison table confounds the two: RAOD is YOLOX-Nano (1M) on
# linear HDR, the torchvision rows are 38-44M on tone-mapped input. Two things
# differ at once.
#
# This holds the front end fixed (RAOD's own Adaptive_Module, initialised from
# their best-day_night checkpoint) and the input fixed (linear HDR), and changes
# only the detector behind it. Same 2-class deduplicated split RAOD was scored
# on: 2307 train / 576 test, Pedestrian and Car.
#
#   YOLOX-Nano   1M   27.0 mAP   <- already measured, RAOD's own model
#   RetinaNet   38M    ?
#   FasterRCNN  44M    ?
#
# Reinhard + FasterRCNN on the same split scores 43.0, so:
#   result near 43 -> capacity was the whole story, the module is fine
#   result near 27 -> the module is the ceiling, not the detector
PY="/c/Users/OGAM/miniconda3/envs/hs-ml/python.exe"
ANN="D:/Data/HDR/hdr4rtt_rod/annotations"
cd "D:/Codes/HDR/Sana/hdr4rtt_rod"
for arch in retinanet fasterrcnn; do
  echo "##### TRAIN tmm + $arch (2-class ROD split) #####"
  "$PY" train_frontend.py --arm tmm --arch "$arch" --epochs 10 --batch 4 --workers 4 \
    --tag rod2cls \
    --train_ann "$ANN/hdr4rtt_rod_dedup_train.json" \
    --val_ann   "$ANN/hdr4rtt_rod_dedup_test.json" \
    || { echo "##### TRAIN FAILED $arch #####"; continue; }
  echo "##### EVAL tmm + $arch #####"
  "$PY" eval_frontend.py --arm tmm --arch "$arch" --batch 4 --tag rod2cls \
    --ann "$ANN/hdr4rtt_rod_dedup_test.json" || echo "##### EVAL FAILED $arch #####"
done
echo "##### CAPACITY RUNS COMPLETE #####"
