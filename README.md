# DrishtiMitra — MATLAB/Simulink screening pipeline

This repository is a MATLAB-first implementation for SIH26038 (MathWorks). It uses Image Processing, Computer Vision, Deep Learning, Medical Imaging, Statistics and Machine Learning Toolboxes, and Simulink. No Python inference pipeline is hidden in this project. Optional ONNX import is performed by Deep Learning Toolbox in `trainDRClassifier.m`.

## Status and evidence policy

The pipeline code and synthetic smoke test are included. **There are currently no real-dataset benchmark values in this repository.** Synthetic results are strictly execution checks and are never performance evidence. `docs/benchmark_results.md` is the sole metric record and remains explicitly unmeasured until registered datasets are supplied and benchmark scripts are run in MATLAB.

MATLAB was not detected in the build environment, so `ScreeningWorkflow.slx` and `DrishtiMitraApp.mlapp` cannot be truthfully generated or validated here. `simulink/buildScreeningWorkflow.m` creates the model when run in Simulink; `app/buildDrishtiMitraApp.m` creates the App Designer artifact when run in MATLAB. Both requirements are tracked as in progress rather than substituted with look-alikes.

## Required dataset layouts

Place data outside source control, under `data/raw`:

```
data/raw/
  DRIVE/{training,test}/{images,1st_manual,mask}/
  IDRiD/{A. Segmentation, B. Disease Grading}/
  APTOS2019/{train.csv,train_images}/
  Messidor2/{images,labels.csv}/
```

The precise registered release structure varies. Set paths in the benchmark scripts; they fail loudly when ground truth is absent. DRIVE and IDRiD require their respective access processes; Messidor-2 requires its data-use approval. The project cannot submit registrations or data-use agreements on a participant's behalf.

## Run order

```matlab
cd DrishtiMitra_MATLAB
addpath(genpath('src'))
run('tests/validate_pipeline.m')           % synthetic smoke checks only
run('simulink/buildScreeningWorkflow.m')   % creates ScreeningWorkflow.slx in Simulink
run('app/buildDrishtiMitraApp.m')          % creates DrishtiMitraApp.mlapp in App Designer
```

For real evaluation, configure dataset paths in `benchmarks/run_real_benchmarks.m`; never copy its results into a pitch without preserving its MATLAB version, hardware, dataset split, and timestamp.

## Toolbox map

Image Processing Toolbox drives morphology, CLAHE, masks, and measurements. Computer Vision Toolbox supplies circle finding and feature utilities. Deep Learning Toolbox provides ONNX import, U-Net/DeepLab construction, and `gradcam`. Statistics and Machine Learning Toolbox provides SVMs and calibration. Medical Imaging Toolbox is used where available for medical-image metadata/volume interoperability. Simulink models service flow and queues.

See `docs/requirements_traceability.md` for complete traceability and `docs/open_questions_and_assumptions.md` before training or presenting.
