#!/bin/bash
# Arm B (1280 input) with the anchor ASPECT RATIOS retuned, sizes left alone.
#
# Measured on the 19,748 training boxes, as the share that can reach IoU 0.5
# with some anchor -- below that RetinaNet cannot assign a positive anchor, so
# it is a ceiling on what any amount of training can recover:
#
#   baseline 800, default ratios          75.3%   (small 45.7)
#   arm B 1280, default ratios            82.9%   (small 58.0)
#   arm B 1280, sizes x1.6                75.3%   (small  0.0)  <- rejected
#   arm B 1280, ratios 1,2.5,5            94.6%   (small 62.1)
#   arm B 1280, ratios 0.5,1.6,6.4        96.0%   (small 62.7)
#
# Scaling the sizes up is what breaks small objects: the smallest anchor moves
# from 32px to 51px and nothing under 32^2 area can reach 0.5 against it. The
# real mismatch was never the sizes, it is that 85% of these objects are taller
# than wide (median height/width 2.39) while the defaults top out at 2.0.
PY="/c/Users/OGAM/miniconda3/envs/hs-ml/python.exe"
ANN="D:/Data/HDR/hdr4rtt_voc20/annotations"
cd "D:/Codes/HDR/Sana/hdr4rtt_rod"
for spec in "ratio_round:1.0,2.5,5.0" "ratio_fit:0.5,1.6,6.4"; do
  tag="${spec%%:*}"; ratios="${spec#*:}"
  echo "##### TRAIN $tag  ratios=$ratios  sizes=default #####"
  "$PY" train_frontend.py --arm reinhard --arch retinanet --epochs 10 --batch 4 \
    --workers 4 --tag "$tag" --min_size 1280 --max_size 2133 \
    --anchor_ratios "$ratios" \
    --train_ann "$ANN/hdr4rtt_voc20_dedup_train.json" \
    --val_ann "$ANN/hdr4rtt_voc20_dedup_test.json" \
    || { echo "##### TRAIN FAILED $tag #####"; continue; }
  echo "##### EVAL $tag #####"
  "$PY" eval_frontend.py --arm reinhard --arch retinanet --batch 4 --tag "$tag" \
    --ann "$ANN/hdr4rtt_voc20_dedup_test.json" || echo "##### EVAL FAILED $tag #####"
done
echo "##### ANCHOR RUNS COMPLETE #####"
