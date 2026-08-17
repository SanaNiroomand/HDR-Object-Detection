# Trains and evaluates every front end against the SAME detector, one after the
# other. Each arm is independent: if one fails the loop continues, so a single
# failure does not cost the whole sweep.
#
# Batch 4 for every arm. Measured peak memory: 3.4 GB for the detector alone and
# 13.1 GB with RAOD's Adaptive_Module in front, against a 17.1 GB card. Windows
# does not report exceeding VRAM -- it pages to system RAM and silently runs ~8x
# slower -- so these were measured rather than assumed.

$py   = "C:\Users\OGAM\miniconda3\envs\hs-ml\python.exe"
$dir  = "D:\Codes\HDR\Sana\hdr4rtt_rod"
$arch = "retinanet"
$arms = @("gamma", "reinhard", "durand", "log", "hdr", "tmm")

Set-Location $dir
foreach ($arm in $arms) {
    Write-Output ""
    Write-Output "##### TRAIN $arch/$arm #####"
    & $py train_frontend.py --arm $arm --arch $arch --epochs 10 --batch 4 --workers 4
    if ($LASTEXITCODE -ne 0) {
        Write-Output "##### TRAIN FAILED for $arm (exit $LASTEXITCODE), continuing #####"
        continue
    }
    Write-Output "##### EVAL $arch/$arm #####"
    & $py eval_frontend.py --arm $arm --arch $arch --batch 4
    if ($LASTEXITCODE -ne 0) {
        Write-Output "##### EVAL FAILED for $arm (exit $LASTEXITCODE) #####"
    }
}
Write-Output ""
Write-Output "##### SWEEP COMPLETE #####"
