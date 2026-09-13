# Benchmark results — authoritative metric ledger

Status: **no real benchmark has been run**. The current build environment has no MATLAB runtime and no registered DRIVE, IDRiD, APTOS 2019, or Messidor-2 data. The earlier PyTorch prototype's QWK 0.9156, accuracy 84.97%, and referable sensitivity 83.33% are reference-only and are not MATLAB implementation results.

| Module / dataset | Metric | Result | Run evidence |
|---|---:|---:|---|
| DR grading / APTOS 2019 | QWK | Unmeasured | Requires held-out MATLAB run |
| Referable DR / APTOS + Messidor-2 | sensitivity / specificity / ROC | Unmeasured | `computeSensSpec.m` emits threshold and ROC |
| Vessels / DRIVE | Dice / sensitivity / specificity | Unmeasured | Requires DRIVE ground truth |
| Microaneurysms / IDRiD | per-lesion sensitivity / precision | Unmeasured | Requires IDRiD masks |
| Exudates / IDRiD | Dice / sensitivity / specificity | Unmeasured | Requires IDRiD masks |
| Hemorrhages / IDRiD | Dice / sensitivity / specificity | Unmeasured | Requires IDRiD masks |
| Neovascularization | sensitivity / specificity | Unmeasured | Requires labelled PDR split |
| Classification-only vs Grad-CAM vs lesion fusion | QWK / sens / spec / AUC | Unmeasured | Requires common held-out split |
| Ophthalmologist mock review | seconds / case | Unmeasured | Must be timed with clinician participants |
| MATLAB pipeline latency | seconds / image | Unmeasured | Record CPU/GPU, MATLAB release, batch size |

## Required operating-point reporting

`computeSensSpec.m` chooses the ROC point nearest `(sensitivity=0.90, specificity=0.85)` unless a held-out threshold is supplied. It reports the actual confusion matrix, AUC, ROC coordinates, threshold, sensitivity, specificity, and whether both targets are met. Threshold selection trades errors along the ROC; it does not guarantee both targets. If the target is missed, improve the fused model, data coverage, loss weighting, or ensemble and remeasure.

Every completed row must include dataset release, split, seed, MATLAB/toolbox version, model checksum, hardware, date, and a link/path to the generated ROC/metrics artifact. Never populate this ledger from smoke-test data.

Note: the official Messidor-2 download does not itself provide DR ground-truth annotations. Any Messidor-2 metric must identify its permitted independent annotation source and matching method.

Use `benchmarks/benchmarkPipelineLatency.m` on representative real images to record p50/p95 latency before setting `scenario.measuredPipelineSeconds`; `runDistrictScenario.m` refuses to provide a staffing/bottleneck conclusion until that input and a measured referable rate exist.
