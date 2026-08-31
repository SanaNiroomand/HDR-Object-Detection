# Re-runs the headline (2-class) table on the DEDUPLICATED split.
#
# The published headline table was measured on a split that still contained the
# near-duplicate S1 frames, so its absolute values are inflated the same way the
# tone-mapping table was before that was re-run. This puts both tables on the
# same footing.
#
# Five rows: two zero-shot evaluations that need no training, and two
# fine-tuning runs. RAOD uses batch 4 (batch 8 exceeds a 16 GB card and Windows
# pages instead of reporting it).

$py    = "C:\Users\OGAM\miniconda3\envs\hs-ml\python.exe"
$dir   = "D:\Codes\HDR\Sana\hdr4rtt_rod"
$raod  = "D:\Codes\HDR\Sana\RAOD\RAOD"
$ann   = "D:\Data\HDR\hdr4rtt_rod\annotations\hdr4rtt_rod_dedup_test.json"
$train = "D:\Data\HDR\hdr4rtt_rod\annotations\hdr4rtt_rod_dedup_train.json"
$npy   = "D:\Data\HDR\hdr4rtt_rod\images"
$ldr   = "D:\Data\HDR\hdr4rtt_ldr\images"
$out   = "D:\Data\HDR\hdr4rtt_rod\headline_dedup"
New-Item -ItemType Directory -Force -Path $out | Out-Null

Set-Location $dir

Write-Output "##### ROW 1: RAOD zero-shot #####"
& $py eval_rod.py --ann $ann --img_dir $npy `
    --ckpt "$raod\pre-trained\best-day_night.pth" --cfg "$raod\cfg_small.py" `
    --gains 1.0 --batch 4 --out_json "$out\raod_zeroshot.json"

Write-Output "##### ROW 3: RetinaNet zero-shot (tone-mapped) #####"
& $py eval_torchvision.py --ann $ann --img_dir $ldr --arch retinanet --weights coco `
    --batch 4 --out_json "$out\retinanet_zeroshot.json"

Write-Output "##### ROW 4: Faster R-CNN zero-shot (tone-mapped) #####"
& $py eval_torchvision.py --ann $ann --img_dir $ldr --arch fasterrcnn --weights coco `
    --batch 4 --out_json "$out\fasterrcnn_zeroshot.json"

Write-Output "##### ROW 5: Faster R-CNN fine-tuned (tone-mapped) #####"
& $py train_torchvision.py --arch fasterrcnn --epochs 12 --batch 2 `
    --train_ann $train --val_ann $ann --img_dir $ldr `
    --out_dir "$dir\tv_runs_dedup"
if ($LASTEXITCODE -eq 0) {
    & $py eval_torchvision.py --ann $ann --img_dir $ldr --arch fasterrcnn `
        --weights "$dir\tv_runs_dedup\fasterrcnn\last.pth" `
        --batch 4 --out_json "$out\fasterrcnn_finetuned.json"
} else {
    Write-Output "##### ROW 5 TRAIN FAILED (exit $LASTEXITCODE) #####"
}

Write-Output "##### ROW 2: RAOD fine-tuned (longest, last) #####"
Set-Location $raod
& $py main.py -f "$dir\cfg_hdr4rtt_rod_dedup.py" -d 1 -b 4 --fp16 `
    -c "$raod\pre-trained\best-day_night.pth"
if ($LASTEXITCODE -eq 0) {
    Set-Location $dir
    & $py eval_rod.py --ann $ann --img_dir $npy `
        --ckpt "$dir\cfg_hdr4rtt_rod_dedup\best_ckpt.pth" `
        --cfg "$dir\cfg_hdr4rtt_rod_dedup.py" `
        --gains 1.0 --batch 4 --out_json "$out\raod_finetuned.json"
} else {
    Write-Output "##### ROW 2 TRAIN FAILED (exit $LASTEXITCODE) #####"
}

Write-Output ""
Write-Output "##### HEADLINE DEDUP COMPLETE #####"
