# Open questions and assumptions

| Question | Current answer / action |
|---|---|
| MATLAB GPU available? | Unknown. `gpuDeviceCount` must be run on the intended training machine, along with Parallel Computing Toolbox availability. Until then the supported decision is ONNX import for baseline inference; native retraining is not assumed. |
| Reuse or retrain? | Provisional: import the approved prior EfficientNet-B0 ONNX in Deep Learning Toolbox, then fine-tune/retrain natively only if GPU/data/time permit. The final decision must be recorded with model checksum and held-out benchmark. |
| Submission deadline | Not supplied. Add the actual local date/time before scope planning; current work cannot infer it. |
| Latency hardware | Unknown. Time this MATLAB implementation on the deployment CPU/GPU, report model, RAM, MATLAB release, GPU/batch size, and the p50/p95 latency. `measuredPipelineSeconds` is deliberately `NaN` until then. |
| Dataset access | Participant must complete the applicable DRIVE/IDRiD access process and accept the Messidor-2 terms. The repository contains no data and cannot submit legal/account forms on their behalf. Official access routes: `drive.grand-challenge.org/Download`, `idrid.grand-challenge.org`, and `adcis.net/en/third-party/messidor2/`. |
| Messidor-2 ground truth | The official Messidor-2 provider says its download does not include DR ground-truth annotations. Obtain and document a permitted independent label source before using it for sensitivity/specificity/QWK; `loadMessidor2.m` requires a provenance-labelled `labels.csv` for supervised evaluation. |
| Clinical review claim | Do not state a <30-second review claim until a timed mock review with representative reports and ophthalmologist participants is recorded. |

## Assumptions used by the district scenario

The scenario starts with 100,000 annual patients, 50 CHO centres, one station/centre, 5 images/station-hour, 250 workdays, 8 MB/image, 5 Mbps uplink, and 40% store-and-forward. These are planning inputs, not measured facts; referable rate and MATLAB processing latency remain unset to prevent false staffing conclusions.
