# Retest of the learned tone-mapping arm.
#
# In the first sweep it placed 4th of 6 (mAP 35.4), beating only the no-curve
# arm. Two of the settings used there could have suppressed it rather than the
# method being weak, so each is isolated here against the same baseline:
#
#   fullr   learning rate raised to match the detector (was 10x slower). Tests
#           whether the slow rate stopped it adapting.
#   rand    RAOD's released weights discarded. They were fitted on automotive
#           RAW sensor data; that may be a worse start than random for indoor
#           HDR photography.
#   gain    input scaled by 0.18 so the mean lands near 0.012, the level RAOD's
#           weights were fitted at (this data averages ~0.068, about 6x brighter).
#
# One variable changes per run. Everything else -- data, split, detector,
# schedule, batch size, augmentation -- matches the original arm exactly, so any
# difference is attributable.

$py   = "C:\Users\OGAM\miniconda3\envs\hs-ml\python.exe"
$dir  = "D:\Codes\HDR\Sana\hdr4rtt_rod"
Set-Location $dir

$variants = @(
    @{tag = "fullr"; extra = @("--tmm_lr_mult", "1.0")},
    @{tag = "rand";  extra = @("--tmm_random_init")},
    @{tag = "gain";  extra = @("--input_gain", "0.18")}
)

foreach ($v in $variants) {
    Write-Output ""
    Write-Output "##### TRAIN tmm/$($v.tag) #####"
    & $py train_frontend.py --arm tmm --arch retinanet --epochs 10 --batch 4 `
        --workers 4 --tag $v.tag @($v.extra)
    if ($LASTEXITCODE -ne 0) {
        Write-Output "##### TRAIN FAILED for $($v.tag) (exit $LASTEXITCODE), continuing #####"
        continue
    }
    Write-Output "##### EVAL tmm/$($v.tag) #####"
    & $py eval_frontend.py --arm tmm --arch retinanet --batch 4 --tag $v.tag
    if ($LASTEXITCODE -ne 0) {
        Write-Output "##### EVAL FAILED for $($v.tag) (exit $LASTEXITCODE) #####"
    }
}
Write-Output ""
Write-Output "##### VARIANTS COMPLETE #####"
