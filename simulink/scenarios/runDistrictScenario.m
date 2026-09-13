function conclusion = runDistrictScenario(scenarioFile)
%RUNDISTRICTSCENARIO Calculate bottleneck after measured workflow inputs exist.
if nargin<1, scenarioFile=fullfile(fileparts(mfilename('fullpath')),'districtLevel_100k.mat'); end
load(scenarioFile,'scenario');
assert(~isnan(scenario.measuredPipelineSeconds) && ~isnan(scenario.referableRate), ...
 'DrishtiMitra:ScenarioInputs','Set measuredPipelineSeconds and referableRate from real MATLAB/validation runs before drawing conclusions.');
acq=ImageAcquisitionRate(scenario.choCenters,scenario.stationsPerCenter,scenario.imagesPerStationHour,scenario.workHoursPerDay,scenario.workDaysPerYear);
transferHours=BandwidthConstraint(scenario.annualPatients,scenario.imageMB,scenario.uplinkMbps,scenario.storeForwardFraction);
processing=ProcessingThroughput(scenario.measuredPipelineSeconds,scenario.parallelWorkers,scenario.workHoursPerDay*scenario.workDaysPerYear);
review=ReviewCapacity(scenario.specialists,scenario.reviewSeconds,scenario.workHoursPerDay,scenario.workDaysPerYear);
requiredReviews=scenario.annualPatients*scenario.referableRate;
[~,idx]=min([acq processing review]); names=["acquisition","processing","specialist review"];
conclusion=struct('acquisitionCapacity',acq,'annualTransferHours',transferHours,'processingCapacity',processing, ...
 'reviewCapacity',review,'requiredReviews',requiredReviews,'bottleneck',names(idx), ...
 'specialistsRequired',ceil(requiredReviews/max(eps,review/scenario.specialists)), ...
 'minimumUplinkMbps',scenario.annualPatients*scenario.imageMB*8*(1-scenario.storeForwardFraction)/(scenario.workHoursPerDay*scenario.workDaysPerYear*3600));
end
