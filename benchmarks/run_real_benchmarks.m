%% RUN_REAL_BENCHMARKS Real-dataset benchmark entry point — never synthetic data.
root=fileparts(fileparts(mfilename('fullpath'))); addpath(genpath(fullfile(root,'src')));
datasets=struct('DRIVE',fullfile(root,'data','raw','DRIVE'), ...
 'IDRiD',fullfile(root,'data','raw','IDRiD'),'APTOS2019',fullfile(root,'data','raw','APTOS2019'), ...
 'Messidor2',fullfile(root,'data','raw','Messidor2'));
names=fieldnames(datasets); missing=names(~cellfun(@isfolder,struct2cell(datasets)));
assert(isempty(missing),'DrishtiMitra:DataUnavailable','Registered data absent: %s. No benchmark was run.',strjoin(missing,', '));
error('DrishtiMitra:ConfigureSplit','Dataset loaders and approved train/validation/test split must be configured before real benchmarking. This guard prevents accidental unsupported metric claims.');
