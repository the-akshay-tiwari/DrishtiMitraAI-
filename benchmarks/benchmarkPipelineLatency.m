function latency = benchmarkPipelineLatency(images, classifier, calibration, threshold, repetitions)
%BENCHMARKPIPELINELATENCY Measure actual MATLAB end-to-end inference timing.
arguments; images; classifier=[]; calibration=[]; threshold=.5; repetitions (1,1) double {mustBeInteger,mustBePositive}=5; end
if ischar(images)||isstring(images), images={imread(images)}; end
assert(iscell(images)&&~isempty(images),'DrishtiMitra:LatencyImages','Supply one or more real representative images.');
times=[];
for i=1:numel(images)
    for r=1:repetitions
        tic; gradeImage(images{i},classifier,calibration,threshold); times(end+1)=toc; %#ok<AGROW>
    end
end
gpu='CPU only'; if gpuDeviceCount>0, d=gpuDevice; gpu=sprintf('%s (%g GB)',d.Name,d.TotalMemory/2^30); end
latency=struct('samplesSeconds',times,'p50Seconds',prctile(times,50),'p95Seconds',prctile(times,95), ...
 'meanSeconds',mean(times),'matlabVersion',version,'gpu',gpu,'timestamp',datetime('now'));
end
