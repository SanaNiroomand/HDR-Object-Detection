# Re-runs the six front ends on the DEDUPLICATED split.
#
# The earlier sweep used the full test set, whose scores are inflated by
# near-duplicate frames -- S1 alone scored AP50 99.2, which is not a real
# generalisation result. Deduplication (Ulas's dedupe_hdr_frames.py, SSIM
# threshold 0.92) reduces 4,080 images to 3,145, removing 72% of S1 and almost
# nothing from S2 and S3.
#
# Everything else is unchanged from the first sweep -- same images on disk, same
# detector, same 10 epochs at batch 4 -- so the two runs differ only in which
# images are in the split. Results land in frontend_runs/<arch>_<arm>_dedup/.

$py   = "C:\Users\OGAM\miniconda3\envs\hs-ml\python.exe"
$dir  = "D:\Codes\HDR\Sana\hdr4rtt_rod"
$ann  = "D:\Data\HDR\hdr4rtt_voc20\annotations"
$arch = "retinanet"
$arms = @("gamma", "reinhard", "durand", "log", "hdr", "tmm")

Set-Location $dir
foreach ($arm in $arms) {
    Write-Output ""
    Write-Output "##### TRAIN $arm (dedup) #####"
    & $py train_frontend.py --arm $arm --arch $arch --epochs 10 --batch 4 --workers 4 `
        --tag dedup `
        --train_ann "$ann\hdr4rtt_voc20_dedup_train.json" `
        --val_ann "$ann\hdr4rtt_voc20_dedup_test.json"
    if ($LASTEXITCODE -ne 0) {
        Write-Output "##### TRAIN FAILED for $arm (exit $LASTEXITCODE), continuing #####"
        continue
    }
    Write-Output "##### EVAL $arm (dedup, whole test set) #####"
    & $py eval_frontend.py --arm $arm --arch $arch --batch 4 --tag dedup `
        --ann "$ann\hdr4rtt_voc20_dedup_test.json"
    # and per source, so the clean source can be reported on its own
    foreach ($g in @("S1","S2","S3")) {
        & $py eval_frontend.py --arm $arm --arch $arch --batch 4 --tag dedup `
            --ann "$ann\hdr4rtt_voc20_dedup_test_$g.json" `
            --out_json "$dir\frontend_runs\${arch}_${arm}_dedup\eval_$g.json" | Out-Null
    }
}
Write-Output ""
Write-Output "##### DEDUP SWEEP COMPLETE #####"
