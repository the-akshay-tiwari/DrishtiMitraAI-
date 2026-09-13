function summary = benchmarkAgainstPublished(results)
%BENCHMARKAGAINSTPUBLISHED Compare actual held-out MATLAB runs; never invents data.
required={'classificationOnly','gradcamOnly','lesionFusion'};
assert(all(isfield(results,required)),'DrishtiMitra:BenchmarkInput','Supply held-out results for all three approaches.');
summary=struct();
for k=1:numel(required)
    r=results.(required{k}); m=computeSensSpec(r.referableScore,r.referableLabel);
    summary.(required{k})=struct('QWK',computeQWK(r.gradeLabel,r.gradePrediction), ...
        'Sensitivity',m.sensitivity,'Specificity',m.specificity,'AUC',m.auc,'Threshold',m.threshold);
end
end
