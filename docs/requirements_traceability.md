# SIH26038 requirements traceability

| PS requirement | Implementation | Status / evidence |
|---|---|---|
| MATLAB + named toolboxes | `src/`, `simulink/`, README toolbox map | Built code; runtime verification pending MATLAB |
| Quality: sharpness, lighting, FOV, actionable rejection | `src/quality/assessQuality.m` | Built; synthetic smoke test only |
| Conservative CLAHE / illumination / denoise | `src/quality/enhanceImage.m` | Built; documents avoidance of aggressive global normalization |
| Optic disc/fovea | `localizeOpticDiscFovea.m` | Built classical baseline; real evaluation pending |
| Vessel classical + U-Net/DeepLab path | `segmentVessels.m`, `trainDRClassifier.m` | Classical built; U-Net training/evaluation in progress pending DRIVE |
| MA + SVM | `detectMicroaneurysms.m` | Built candidate/features/classifier hook; IDRiD evaluation pending |
| Exudates / haemorrhage / neovascularization | associated `src/segmentation` files | Built classical baselines; ground-truth evaluation pending |
| Transfer learning / ONNX / calibration / lesion fusion | `src/grading/` | Built; trained weights and held-out calibration pending |
| Honest sensitivity, specificity, ROC, QWK | `src/evaluation/` | Built; no real metrics claimed |
| Grad-CAM, evidence overlap, clinical report | `src/explainability/` | Built; requires trained network for Grad-CAM |
| Simulink screening workflow + scenario | `simulink/buildScreeningWorkflow.m`, scenarios | In progress: generator ready; needs Simulink to produce/validate `.slx` and measured inputs |
| District bottleneck/staffing conclusion | `simulink/scenarios/runDistrictScenario.m` | In progress: intentionally withheld until latency and referral rate measured |
| App Designer `.mlapp` | `app/buildDrishtiMitraApp.m` | In progress: requires App Designer runtime; no `.m` substitute created |
| Smoke vs real benchmark separation | `tests/validate_pipeline.m`, `benchmarks/run_real_benchmarks.m`, metric ledger | Built |
